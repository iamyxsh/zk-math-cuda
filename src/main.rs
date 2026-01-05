use cudarc::driver::CudaDevice;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dev = CudaDevice::new(0)?;

    println!("GPU device 0 initialised");

    let host_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let dev_data = dev.htod_sync_copy(&host_data)?;
    let roundtrip: Vec<f32> = dev.dtoh_sync_copy(&dev_data)?;

    assert_eq!(host_data, roundtrip);
    println!("htod → dtoh roundtrip: {:?} == {:?}", host_data, roundtrip);

    let zeroed: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(8)?;
    let zeroed_host: Vec<f32> = dev.dtoh_sync_copy(&zeroed)?;
    assert!(zeroed_host.iter().all(|&v| v == 0.0));
    println!("zeroed alloc: {:?}", zeroed_host);

    Ok(())
}

fn has_gpu() -> bool {
    CudaDevice::new(0).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let zeroed: cudarc::driver::CudaSlice<f32> = dev.alloc_zeros::<f32>(64).unwrap();
        let host: Vec<f32> = dev.dtoh_sync_copy(&zeroed).unwrap();
        assert_eq!(host.len(), 64);
        assert!(
            host.iter().all(|&v| v == 0.0),
            "alloc_zeros should return all zeroes"
        );
    }
}
