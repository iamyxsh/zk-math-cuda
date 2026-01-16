use std::ffi::CStr;

extern "C" {
    fn cupti_timing_init() -> i32;
    fn cupti_timing_flush();
    fn cupti_timing_disable();
    fn cupti_timing_reset();
    fn cupti_timing_count() -> i32;
    fn cupti_timing_get(
        idx: i32,
        start_ns: *mut u64,
        end_ns: *mut u64,
        name: *mut i8,
        name_len: i32,
    );
}

#[derive(Debug, Clone)]
pub struct KernelTiming {
    pub name: String,
    pub start_ns: u64,
    pub end_ns: u64,
}

impl KernelTiming {
    pub fn duration_us(&self) -> f64 {
        (self.end_ns - self.start_ns) as f64 / 1_000.0
    }
}

/// RAII wrapper around the CUPTI Activity API timing session.
///
/// On construction: enables CONCURRENT_KERNEL activity recording.
/// On drop: disables it.
pub struct CuptiTimer;

impl CuptiTimer {
    pub fn new() -> Result<Self, String> {
        let rc = unsafe { cupti_timing_init() };
        if rc != 0 {
            return Err(format!("cuptiActivityEnable failed: CUPTI error {}", rc));
        }
        Ok(CuptiTimer)
    }

    /// Flush CUPTI buffers — must be called after kernel + synchronize
    /// before reading results. Triggers buffer_completed callbacks.
    pub fn flush(&self) {
        unsafe { cupti_timing_flush() };
    }

    /// Clear accumulated records without disabling CUPTI.
    pub fn reset(&self) {
        unsafe { cupti_timing_reset() };
    }

    pub fn results(&self) -> Vec<KernelTiming> {
        let count = unsafe { cupti_timing_count() };
        (0..count)
            .map(|i| {
                let mut start = 0u64;
                let mut end = 0u64;
                let mut name_buf = vec![0i8; 256];
                unsafe {
                    cupti_timing_get(
                        i,
                        &mut start,
                        &mut end,
                        name_buf.as_mut_ptr(),
                        256,
                    );
                }
                let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                    .to_string_lossy()
                    .into_owned();
                KernelTiming { name, start_ns: start, end_ns: end }
            })
            .collect()
    }
}

impl Drop for CuptiTimer {
    fn drop(&mut self) {
        unsafe { cupti_timing_disable() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::has_gpu;

    #[test]
    fn test_cupti_timer_init() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let timer = CuptiTimer::new();
        assert!(timer.is_ok(), "CuptiTimer::new should succeed with a GPU present");
    }

    #[test]
    fn test_cupti_results_empty_before_launch() {
        if !has_gpu() { eprintln!("skipping: no GPU available"); return; }
        let timer = CuptiTimer::new().unwrap();
        timer.flush();
        // No kernel launched — result count should be 0
        assert_eq!(timer.results().len(), 0);
    }
}
