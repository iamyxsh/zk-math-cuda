pub mod cupti;
pub mod field;
pub mod poseidon;

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

    // GPU Poseidon with CUPTI timing
    let timer = cupti::CuptiTimer::new()?;
    let gpu_out = poseidon_gpu(&dev, &[input])?;
    timer.flush();

    assert_eq!(gpu_out[0], cpu_out, "GPU output must match CPU");
    println!("poseidon GPU == CPU ✓");

    for t in timer.results() {
        println!("  kernel: {:?}  duration: {:.2} μs", t.name, t.duration_us());
    }

    Ok(())
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
}
