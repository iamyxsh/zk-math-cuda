use std::ffi::CString;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct DeviceProps {
    pub max_threads_per_sm: i32,
    pub max_blocks_per_sm: i32,
    pub max_regs_per_sm: i32,
    pub max_smem_per_sm: i32,
    pub warp_size: i32,
    pub num_sms: i32,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct KernelOccupancy {
    pub max_active_blocks_per_sm: i32,
    pub regs_per_thread: i32,
    pub static_smem_bytes: i32,
    pub theoretical_occupancy: f32,
}

extern "C" {
    fn occupancy_device_props(out: *mut DeviceProps) -> i32;
    fn occupancy_theoretical(
        ptx: *const i8,
        func_name: *const i8,
        block_size: i32,
        dyn_smem: usize,
        out: *mut KernelOccupancy,
    ) -> i32;
}

#[derive(Debug)]
pub struct OccupancyReport {
    pub device: DeviceProps,
    pub kernel: KernelOccupancy,
    pub theoretical_pct: f64,
    pub achieved_pct: f64,
}

pub fn query_device_props() -> Result<DeviceProps, String> {
    let mut props = DeviceProps {
        max_threads_per_sm: 0,
        max_blocks_per_sm: 0,
        max_regs_per_sm: 0,
        max_smem_per_sm: 0,
        warp_size: 0,
        num_sms: 0,
    };
    let rc = unsafe { occupancy_device_props(&mut props) };
    if rc != 0 {
        return Err(format!("occupancy_device_props failed: {}", rc));
    }
    Ok(props)
}

pub fn query_theoretical(
    ptx: &str,
    func_name: &str,
    block_size: i32,
    dyn_smem: usize,
) -> Result<KernelOccupancy, String> {
    let ptx_c = CString::new(ptx).map_err(|e| format!("PTX null byte: {}", e))?;
    let name_c = CString::new(func_name).map_err(|e| format!("func name null: {}", e))?;
    let mut occ = KernelOccupancy {
        max_active_blocks_per_sm: 0,
        regs_per_thread: 0,
        static_smem_bytes: 0,
        theoretical_occupancy: 0.0,
    };
    let rc = unsafe {
        occupancy_theoretical(
            ptx_c.as_ptr(),
            name_c.as_ptr(),
            block_size,
            dyn_smem,
            &mut occ,
        )
    };
    if rc != 0 {
        return Err(format!("occupancy_theoretical failed: {}", rc));
    }
    Ok(occ)
}

/// Compute both theoretical and achieved occupancy for a kernel.
///
/// Theoretical = max warps given the kernel's resource footprint.
/// Achieved = warps that actually get scheduled given the grid size,
/// capped by the theoretical limit.
pub fn compute_occupancy(
    ptx: &str,
    func_name: &str,
    block_size: u32,
    grid_size: u32,
    dyn_smem: usize,
) -> Result<OccupancyReport, String> {
    let device = query_device_props()?;
    let kernel = query_theoretical(ptx, func_name, block_size as i32, dyn_smem)?;

    let theoretical_pct = kernel.theoretical_occupancy as f64 * 100.0;

    let warp_size = device.warp_size as f64;
    let max_warps_per_sm = device.max_threads_per_sm as f64 / warp_size;
    let warps_per_block = (block_size as f64 / warp_size).ceil();

    // How many blocks actually land on each SM (ceiling division,
    // capped by the hardware/register limit).
    let blocks_per_sm = (grid_size as f64 / device.num_sms as f64)
        .ceil()
        .min(kernel.max_active_blocks_per_sm as f64);

    let achieved_pct = if max_warps_per_sm > 0.0 {
        (blocks_per_sm * warps_per_block / max_warps_per_sm) * 100.0
    } else {
        0.0
    };

    Ok(OccupancyReport {
        device,
        kernel,
        theoretical_pct,
        achieved_pct,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::has_gpu;

    #[test]
    fn test_device_props_query() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = cudarc::driver::CudaDevice::new(0).unwrap();
        let props = query_device_props().unwrap();
        assert!(props.num_sms > 0, "SM count must be positive");
        assert_eq!(props.warp_size, 32, "NVIDIA warp size is always 32");
        assert!(props.max_threads_per_sm > 0);
        assert!(props.max_regs_per_sm > 0);
    }

    #[test]
    fn test_theoretical_occupancy_valid() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = cudarc::driver::CudaDevice::new(0).unwrap();
        let ptx = include_str!("../kernels/poseidon.ptx");
        let occ = query_theoretical(ptx, "poseidon_permutation_kernel", 256, 0).unwrap();
        assert!(occ.regs_per_thread > 0, "kernel must use registers");
        assert!(occ.max_active_blocks_per_sm > 0);
        assert!(
            occ.theoretical_occupancy > 0.0 && occ.theoretical_occupancy <= 1.0,
            "theoretical occupancy {:.2} out of range",
            occ.theoretical_occupancy
        );
    }
}
