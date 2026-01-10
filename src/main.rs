pub mod field;
pub mod poseidon;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
use std::sync::Arc;

const VECTOR_ADD_PTX: &str = include_str!("../kernels/vector_add.ptx");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = CudaDevice::new(0)?;
    println!("GPU device 0 initialised");

    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];

    let result = vector_add(&dev, &a, &b)?;
    println!("vector_add({:?}, {:?}) = {:?}", a, b, result);

    // CPU Poseidon demo
    let input = [field::Fp::ZERO; poseidon::T];
    let output = poseidon::poseidon_permutation(&input);
    println!("poseidon([0,0,0]) = {:?}", output.map(|x| x.from_mont()));

    Ok(())
}

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

    let result = dev.dtoh_sync_copy(&c_dev)?;
    Ok(result)
}

fn has_gpu() -> bool {
    CudaDevice::new(0).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Step 1 tests ---

    #[test]
    fn test_gpu_device_init() {
        if !has_gpu() {
            eprintln!("skipping: no GPU available");
            return;
        }
        let dev = CudaDevice::new(0);
        assert!(dev.is_ok(), "CudaDevice::new(0) should succeed");
    }

    #[test]
    fn test_htod_dtoh_roundtrip() {
        if !has_gpu() {
            eprintln!("skipping: no GPU available");
            return;
        }
        let dev = CudaDevice::new(0).unwrap();
        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dev_data = dev.htod_sync_copy(&host).unwrap();
        let result: Vec<f32> = dev.dtoh_sync_copy(&dev_data).unwrap();
        assert_eq!(host, result, "roundtrip should preserve exact values");
    }

    #[test]
    fn test_device_alloc_zeroed() {
        if !has_gpu() {
            eprintln!("skipping: no GPU available");
            return;
        }
        let dev = CudaDevice::new(0).unwrap();
        let zeroed: CudaSlice<f32> = dev.alloc_zeros::<f32>(64).unwrap();
        let host: Vec<f32> = dev.dtoh_sync_copy(&zeroed).unwrap();
        assert_eq!(host.len(), 64);
        assert!(
            host.iter().all(|&v| v == 0.0),
            "alloc_zeros should return all zeroes"
        );
    }

    // --- Step 2 tests ---

    #[test]
    fn test_vector_add_basic() {
        if !has_gpu() {
            eprintln!("skipping: no GPU available");
            return;
        }
        let dev = CudaDevice::new(0).unwrap();
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let result = vector_add(&dev, &a, &b).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_vector_add_large() {
        if !has_gpu() {
            eprintln!("skipping: no GPU available");
            return;
        }
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
        if !has_gpu() {
            eprintln!("skipping: no GPU available");
            return;
        }
        let dev = CudaDevice::new(0).unwrap();
        let n = 1000; // not a multiple of block_dim=256
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        let result = vector_add(&dev, &a, &b).unwrap();
        assert_eq!(result.len(), n);
        for i in 0..n {
            assert_eq!(result[i], (i + i * 2) as f32, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_vector_add_empty() {
        if !has_gpu() {
            eprintln!("skipping: no GPU available");
            return;
        }
        let dev = CudaDevice::new(0).unwrap();
        let result = vector_add(&dev, &[], &[]).unwrap();
        assert!(result.is_empty());
    }
}
