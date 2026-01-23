use std::ffi::CStr;

extern "C" {
    fn stall_counters_init() -> i32;
    fn stall_counters_read();
    fn stall_counters_disable();
    fn stall_counters_count() -> i32;
    fn stall_counters_get(idx: i32, name: *mut i8, name_len: i32, count: *mut u64);
}

#[derive(Debug, Clone)]
pub struct StallCounter {
    pub name: String,
    pub count: u64,
}

/// RAII guard that enables CUPTI warp-stall event collection.
///
/// Must be constructed **after** a CUDA context is active (i.e. after
/// `CudaDevice::new`).  On drop, the event group is disabled.
pub struct StallProfiler;

impl StallProfiler {
    pub fn new() -> Result<Self, String> {
        let rc = unsafe { stall_counters_init() };
        match rc {
            0   => Ok(StallProfiler),
            -1  => Err("no active CUDA context".to_string()),
            -2  => Err("could not query CUDA device from context".to_string()),
            -3  => Err("no stall events available on this GPU/driver".to_string()),
            n   => Err(format!("CUPTI event group error: code {}", n)),
        }
    }

    /// Read accumulated stall counts after a kernel launch + synchronize.
    pub fn read(&self) -> Vec<StallCounter> {
        unsafe { stall_counters_read() };
        let n = unsafe { stall_counters_count() };
        (0..n)
            .map(|i| {
                let mut name_buf = vec![0i8; 64];
                let mut cnt = 0u64;
                unsafe {
                    stall_counters_get(i, name_buf.as_mut_ptr(), 64, &mut cnt);
                }
                let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                StallCounter { name, count: cnt }
            })
            .collect()
    }
}

impl Drop for StallProfiler {
    fn drop(&mut self) {
        unsafe { stall_counters_disable() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::has_gpu;

    #[test]
    fn test_stall_profiler_init() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = cudarc::driver::CudaDevice::new(0).unwrap();
        let p = StallProfiler::new();
        assert!(p.is_ok(), "StallProfiler::new failed: {:?}", p.err());
    }

    #[test]
    fn test_stall_profiler_read_returns_records() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let _dev = cudarc::driver::CudaDevice::new(0).unwrap();
        let p = StallProfiler::new().unwrap();
        let records = p.read();
        assert!(!records.is_empty(), "expect at least one stall event registered");
    }
}
