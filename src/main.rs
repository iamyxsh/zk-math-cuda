pub mod cupti;
pub mod field;
pub mod poseidon;
pub mod occupancy;
pub mod stall_counters;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use field::Fp;
use std::sync::Arc;

const VECTOR_ADD_PTX: &str = include_str!("../kernels/vector_add.ptx");
// Compile first: nvcc -ptx -arch=sm_XX kernels/poseidon.cu -o kernels/poseidon.ptx
const POSEIDON_PTX: &str = include_str!("../kernels/poseidon.ptx");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = CudaDevice::new(0)?;
    println!("GPU device 0 initialised");

    // vector add demo
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];
    let result = vector_add(&dev, &a, &b)?;
    println!("vector_add({:?}, {:?}) = {:?}", a, b, result);

    // CPU Poseidon
    let input = [Fp::ZERO; poseidon::T];
    let cpu_out = poseidon::poseidon_permutation(&input);

    // GPU Poseidon correctness check
    let timer = cupti::CuptiTimer::new()?;
    let gpu_out = poseidon_gpu(&dev, &[input])?;
    timer.flush();
    assert_eq!(gpu_out[0], cpu_out, "GPU output must match CPU");
    println!("poseidon GPU == CPU ✓");

    // Multi-launch timing variance
    let stats = run_multi_launch(&dev, 20, 256)?;
    println!(
        "\nposeidon 20 launches  batch=256\n  min {:>9.2} μs\n  max {:>9.2} μs\n  mean {:>8.2} μs\n  stddev {:>6.2} μs",
        stats.min_us, stats.max_us, stats.mean_us, stats.stddev_us
    );

    // Warp stall counters
    match profile_poseidon_stalls(&dev, 256) {
        Ok(stalls) => {
            println!("\nposeidon stall breakdown (batch=256):");
            for s in &stalls {
                println!("  {:<35} {}", s.name, s.count);
            }
        }
        Err(e) => eprintln!("stall profiling unavailable: {}", e),
    }

    // Occupancy metrics
    let block_size = 256u32;
    let batch = 256usize;
    let grid_size = (batch as u32 + block_size - 1) / block_size;
    match occupancy::compute_occupancy(
        POSEIDON_PTX, "poseidon_permutation_kernel",
        block_size, grid_size, 0,
    ) {
        Ok(report) => {
            println!("\noccupancy (block={}, grid={}):", block_size, grid_size);
            println!("  SMs              {}", report.device.num_sms);
            println!("  regs/thread      {}", report.kernel.regs_per_thread);
            println!("  max blocks/SM    {}", report.kernel.max_active_blocks_per_sm);
            println!("  theoretical      {:.1}%", report.theoretical_pct);
            println!("  achieved         {:.1}%", report.achieved_pct);
        }
        Err(e) => eprintln!("occupancy query failed: {}", e),
    }

    Ok(())
}

// --- Warp stall profiling ---

fn profile_poseidon_stalls(
    dev: &Arc<CudaDevice>,
    batch_size: usize,
) -> Result<Vec<stall_counters::StallCounter>, Box<dyn std::error::Error>> {
    let states: Vec<[Fp; poseidon::T]> = (0..batch_size)
        .map(|i| [Fp::from_u64(i as u64), Fp::ZERO, Fp::ZERO])
        .collect();

    let profiler = stall_counters::StallProfiler::new()?;
    poseidon_gpu(dev, &states)?;
    Ok(profiler.read())
}

// --- Multi-launch timing ---

#[derive(Debug)]
pub struct TimingStats {
    pub samples: Vec<f64>,
    pub min_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
    pub stddev_us: f64,
}

impl TimingStats {
    pub fn from_durations(samples: Vec<f64>) -> Self {
        assert!(!samples.is_empty());
        let n = samples.len() as f64;
        let min_us = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_us = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_us = samples.iter().sum::<f64>() / n;
        let variance = samples.iter().map(|&x| (x - mean_us).powi(2)).sum::<f64>() / n;
        TimingStats { samples, min_us, max_us, mean_us, stddev_us: variance.sqrt() }
    }
}

fn run_multi_launch(
    dev: &Arc<CudaDevice>,
    n_launches: usize,
    batch_size: usize,
) -> Result<TimingStats, Box<dyn std::error::Error>> {
    let states: Vec<[Fp; poseidon::T]> = (0..batch_size)
        .map(|i| [Fp::from_u64(i as u64), Fp::ZERO, Fp::ZERO])
        .collect();

    let timer = cupti::CuptiTimer::new()?;

    for _ in 0..n_launches {
        poseidon_gpu(dev, &states)?;
    }
    timer.flush();

    let records = timer.results();
    if records.len() != n_launches {
        return Err(format!(
            "expected {} timing records, got {}",
            n_launches,
            records.len()
        )
        .into());
    }

    let durations: Vec<f64> = records.iter().map(|t| t.duration_us()).collect();
    Ok(TimingStats::from_durations(durations))
}

// --- Poseidon GPU launcher ---

// Each Fp([u64; 4]) becomes 8 × u32 (split each u64 into lo/hi, little-endian).
// Matches the FpGpu layout in poseidon.cu.
fn fp_to_gpu_limbs(fp: &Fp) -> [u32; 8] {
    let mut out = [0u32; 8];
    for (i, &limb) in fp.0.iter().enumerate() {
        out[i * 2] = limb as u32;
        out[i * 2 + 1] = (limb >> 32) as u32;
    }
    out
}

fn fp_from_gpu_limbs(limbs: &[u32; 8]) -> Fp {
    Fp([
        (limbs[1] as u64) << 32 | limbs[0] as u64,
        (limbs[3] as u64) << 32 | limbs[2] as u64,
        (limbs[5] as u64) << 32 | limbs[4] as u64,
        (limbs[7] as u64) << 32 | limbs[6] as u64,
    ])
}

fn poseidon_gpu(
    dev: &Arc<CudaDevice>,
    states: &[[Fp; poseidon::T]],
) -> Result<Vec<[Fp; poseidon::T]>, Box<dyn std::error::Error>> {
    if states.is_empty() {
        return Ok(vec![]);
    }

    if dev
        .get_func("poseidon_mod", "poseidon_permutation_kernel")
        .is_none()
    {
        dev.load_ptx(
            cudarc::driver::Ptx::from_src(POSEIDON_PTX),
            "poseidon_mod",
            &["poseidon_permutation_kernel"],
        )?;
    }
    let f = dev
        .get_func("poseidon_mod", "poseidon_permutation_kernel")
        .unwrap();

    let batch = states.len();
    const LIMBS: usize = 8; // u32 limbs per Fp

    // Pack: [state0_fp0_limbs, state0_fp1_limbs, state0_fp2_limbs, state1_fp0_limbs, ...]
    let input_flat: Vec<u32> = states
        .iter()
        .flat_map(|state| state.iter().flat_map(|fp| fp_to_gpu_limbs(fp)))
        .collect();

    let input_dev = dev.htod_sync_copy(&input_flat)?;
    let output_dev = dev.alloc_zeros::<u32>(batch * poseidon::T * LIMBS)?;

    let block_dim = 256u32;
    let grid_dim = (batch as u32 + block_dim - 1) / block_dim;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe { f.launch(cfg, (&input_dev, &output_dev, batch as u32)) }?;

    let output_flat = dev.dtoh_sync_copy(&output_dev)?;

    let results = output_flat
        .chunks_exact(poseidon::T * LIMBS)
        .map(|state_raw| {
            let mut state = [Fp::ZERO; poseidon::T];
            for j in 0..poseidon::T {
                let chunk: &[u32; 8] = state_raw[j * LIMBS..(j + 1) * LIMBS]
                    .try_into()
                    .unwrap();
                state[j] = fp_from_gpu_limbs(chunk);
            }
            state
        })
        .collect();

    Ok(results)
}

// --- Vector add ---

fn vector_add(
    dev: &Arc<CudaDevice>,
    a: &[f32],
    b: &[f32],
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    assert_eq!(a.len(), b.len());
    let n = a.len();

    if n == 0 {
        return Ok(vec![]);
    }

    dev.load_ptx(
        cudarc::driver::Ptx::from_src(VECTOR_ADD_PTX),
        "vector_add_mod",
        &["vector_add"],
    )?;

    let f = dev.get_func("vector_add_mod", "vector_add").unwrap();
    let a_dev = dev.htod_sync_copy(a)?;
    let b_dev = dev.htod_sync_copy(b)?;
    let c_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(n)?;

    let block_dim = 256u32;
    let grid_dim = ((n as u32) + block_dim - 1) / block_dim;

    let cfg = LaunchConfig {
        grid_dim: (grid_dim, 1, 1),
        block_dim: (block_dim, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe { f.launch(cfg, (&a_dev, &b_dev, &c_dev, n as u32)) }?;

    Ok(dev.dtoh_sync_copy(&c_dev)?)
}

fn has_gpu() -> bool {
    CudaDevice::new(0).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Step 1 ---

    #[test]
    fn test_gpu_device_init() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        assert!(CudaDevice::new(0).is_ok());
    }

    #[test]
    fn test_htod_dtoh_roundtrip() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dev_data = dev.htod_sync_copy(&host).unwrap();
        let result: Vec<f32> = dev.dtoh_sync_copy(&dev_data).unwrap();
        assert_eq!(host, result);
    }

    #[test]
    fn test_device_alloc_zeroed() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let zeroed: CudaSlice<f32> = dev.alloc_zeros::<f32>(64).unwrap();
        let host: Vec<f32> = dev.dtoh_sync_copy(&zeroed).unwrap();
        assert!(host.iter().all(|&v| v == 0.0));
    }

    // --- Step 2 ---

    #[test]
    fn test_vector_add_basic() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let result = vector_add(&dev, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_add_large() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let n = 1_000_000;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let result = vector_add(&dev, &a, &b).unwrap();
        for i in 0..n {
            assert_eq!(result[i], 2.0 * i as f32, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_vector_add_misaligned_size() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let n = 1000;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let result = vector_add(&dev, &a, &b).unwrap();
        for i in 0..n {
            assert_eq!(result[i], (i + i * 2) as f32, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_vector_add_empty() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        assert!(vector_add(&dev, &[], &[]).unwrap().is_empty());
    }

    // --- Step 4: GPU vs CPU correctness ---

    #[test]
    fn test_poseidon_gpu_matches_cpu_zero() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let input = [Fp::ZERO; poseidon::T];
        let cpu = poseidon::poseidon_permutation(&input);
        let gpu = poseidon_gpu(&dev, &[input]).unwrap();
        assert_eq!(gpu[0], cpu, "GPU must match CPU bit-for-bit on zero input");
    }

    #[test]
    fn test_poseidon_gpu_matches_cpu_nonzero() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let input = [Fp::from_u64(1), Fp::from_u64(2), Fp::from_u64(3)];
        let cpu = poseidon::poseidon_permutation(&input);
        let gpu = poseidon_gpu(&dev, &[input]).unwrap();
        assert_eq!(gpu[0], cpu, "GPU must match CPU bit-for-bit on [1,2,3]");
    }

    #[test]
    fn test_poseidon_gpu_batch() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let states: Vec<[Fp; poseidon::T]> = (0u64..64)
            .map(|i| [Fp::from_u64(i), Fp::from_u64(i + 1), Fp::ZERO])
            .collect();
        let gpu_out = poseidon_gpu(&dev, &states).unwrap();
        for (i, (state, gpu)) in states.iter().zip(gpu_out.iter()).enumerate() {
            let cpu = poseidon::poseidon_permutation(state);
            assert_eq!(gpu, &cpu, "batch mismatch at index {}", i);
        }
    }

    #[test]
    fn test_poseidon_gpu_empty_batch() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let result = poseidon_gpu(&dev, &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_fp_limb_roundtrip() {
        let fp = Fp::from_u64(0xdeadbeef_cafebabe);
        let limbs = fp_to_gpu_limbs(&fp);
        let recovered = fp_from_gpu_limbs(&limbs);
        assert_eq!(fp, recovered);
    }

    // --- Step 5: CUPTI Activity API timing ---

    #[test]
    fn test_cupti_timing_poseidon() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let timer = cupti::CuptiTimer::new().expect("CUPTI init failed");

        poseidon_gpu(&dev, &[[Fp::ZERO; poseidon::T]]).unwrap();
        timer.flush();

        let results = timer.results();
        assert!(!results.is_empty(), "expected at least one kernel timing record");

        let t = &results[0];
        assert!(t.end_ns > t.start_ns, "end must be after start");
        assert!(t.duration_us() > 0.0, "duration must be positive");
    }

    #[test]
    fn test_cupti_timing_duration_plausible() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let timer = cupti::CuptiTimer::new().unwrap();

        poseidon_gpu(&dev, &[[Fp::ZERO; poseidon::T]]).unwrap();
        timer.flush();

        let results = timer.results();
        assert!(!results.is_empty());
        let us = results[0].duration_us();
        // Sanity: Poseidon permutation should take between 1μs and 100ms
        assert!(us > 1.0,     "duration {} μs suspiciously short", us);
        assert!(us < 100_000.0, "duration {} μs suspiciously long", us);
    }

    #[test]
    fn test_cupti_timing_reset() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let timer = cupti::CuptiTimer::new().unwrap();

        poseidon_gpu(&dev, &[[Fp::ZERO; poseidon::T]]).unwrap();
        timer.flush();
        assert!(!timer.results().is_empty());

        timer.reset();
        assert_eq!(timer.results().len(), 0, "reset should clear all records");
    }

    // --- Step 6: multi-launch timing variance ---

    #[test]
    fn test_multi_launch_count() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let stats = run_multi_launch(&dev, 10, 1).unwrap();
        assert_eq!(stats.samples.len(), 10, "one timing record per launch");
    }

    #[test]
    fn test_multi_launch_all_positive() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let stats = run_multi_launch(&dev, 5, 64).unwrap();
        for (i, &us) in stats.samples.iter().enumerate() {
            assert!(us > 0.0, "launch {} duration must be positive, got {}", i, us);
        }
    }

    #[test]
    fn test_multi_launch_stats_invariants() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let stats = run_multi_launch(&dev, 10, 256).unwrap();
        assert!(stats.min_us <= stats.mean_us, "min <= mean");
        assert!(stats.mean_us <= stats.max_us, "mean <= max");
        assert!(stats.stddev_us >= 0.0,        "stddev >= 0");
        assert!(stats.mean_us > 0.0,           "mean > 0");
    }

    #[test]
    fn test_multi_launch_larger_batch_is_slower() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let small = run_multi_launch(&dev, 5, 1).unwrap();
        let large = run_multi_launch(&dev, 5, 4096).unwrap();
        assert!(
            large.mean_us > small.mean_us,
            "batch=4096 mean {:.2}μs should exceed batch=1 mean {:.2}μs",
            large.mean_us, small.mean_us
        );
    }

    // --- Step 7: warp stall counters ---

    #[test]
    fn test_stall_profiler_creates_after_context() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = CudaDevice::new(0).unwrap();
        let p = stall_counters::StallProfiler::new();
        assert!(p.is_ok(), "StallProfiler::new failed: {:?}", p.err());
    }

    #[test]
    fn test_stall_counters_returns_known_events() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let p = stall_counters::StallProfiler::new().unwrap();
        poseidon_gpu(&dev, &[[Fp::ZERO; poseidon::T]]).unwrap();
        let stalls = p.read();
        assert!(!stalls.is_empty(), "at least one stall event expected");
        for s in &stalls {
            assert!(
                s.name.starts_with("stall_"),
                "unexpected event name: '{}'", s.name
            );
        }
    }

    #[test]
    fn test_stall_memory_dependency_present() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let p = stall_counters::StallProfiler::new().unwrap();
        poseidon_gpu(&dev, &[[Fp::ZERO; poseidon::T]]).unwrap();
        let stalls = p.read();
        let found = stalls.iter().any(|s| s.name == "stall_memory_dependency");
        assert!(found, "stall_memory_dependency must be present; got: {:?}",
            stalls.iter().map(|s| &s.name).collect::<Vec<_>>());
    }

    #[test]
    fn test_stall_counts_nonzero_large_batch() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let dev = CudaDevice::new(0).unwrap();
        let stalls = profile_poseidon_stalls(&dev, 1024).unwrap();
        let total: u64 = stalls.iter().map(|s| s.count).sum();
        assert!(total > 0, "expected non-zero stall counts for batch=1024, got 0");
    }

    // --- Step 8: occupancy metrics ---

    #[test]
    fn test_device_props_sane() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = CudaDevice::new(0).unwrap();
        let props = occupancy::query_device_props().unwrap();
        assert!(props.num_sms >= 1 && props.num_sms <= 256,
            "SM count {} out of range", props.num_sms);
        assert!(props.max_threads_per_sm >= 1024,
            "max threads/SM {} too low", props.max_threads_per_sm);
    }

    #[test]
    fn test_occupancy_report_consistency() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = CudaDevice::new(0).unwrap();
        let report = occupancy::compute_occupancy(
            POSEIDON_PTX, "poseidon_permutation_kernel", 256, 1, 0,
        ).unwrap();
        // Single block: achieved <= theoretical
        assert!(report.achieved_pct <= report.theoretical_pct + 0.01,
            "achieved {:.1}% should not exceed theoretical {:.1}%",
            report.achieved_pct, report.theoretical_pct);
    }

    #[test]
    fn test_occupancy_achieved_scales_with_grid() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = CudaDevice::new(0).unwrap();
        let small = occupancy::compute_occupancy(
            POSEIDON_PTX, "poseidon_permutation_kernel", 256, 1, 0,
        ).unwrap();
        let large = occupancy::compute_occupancy(
            POSEIDON_PTX, "poseidon_permutation_kernel", 256, 1024, 0,
        ).unwrap();
        assert!(large.achieved_pct >= small.achieved_pct,
            "more blocks should give >= achieved occupancy");
    }

    #[test]
    fn test_occupancy_theoretical_bounded() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = CudaDevice::new(0).unwrap();
        let report = occupancy::compute_occupancy(
            POSEIDON_PTX, "poseidon_permutation_kernel", 256, 256, 0,
        ).unwrap();
        assert!(report.theoretical_pct > 0.0 && report.theoretical_pct <= 100.0,
            "theoretical {:.1}% out of [0, 100]", report.theoretical_pct);
    }
}
