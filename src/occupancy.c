// Occupancy queries via the CUDA driver API.
//
// Theoretical occupancy comes from cuOccupancyMaxActiveBlocksPerMultiprocessor,
// which accounts for register pressure, shared-memory usage, block size, and
// the hardware limits of the target SM.  The PTX is JIT-compiled by the driver
// into a temporary CUmodule so we can query the compiled kernel's resource
// footprint.

#include <cuda.h>
#include <stdint.h>

// Mirrors the Rust-side #[repr(C)] DeviceProps.
typedef struct {
    int max_threads_per_sm;
    int max_blocks_per_sm;
    int max_regs_per_sm;
    int max_smem_per_sm;   // bytes
    int warp_size;
    int num_sms;
} DeviceProps;

// Mirrors the Rust-side #[repr(C)] KernelOccupancy.
typedef struct {
    int   max_active_blocks_per_sm;
    int   regs_per_thread;
    int   static_smem_bytes;
    float theoretical_occupancy;  // 0.0 – 1.0
} KernelOccupancy;

int occupancy_device_props(DeviceProps *out) {
    CUdevice dev;
    CUresult r = cuCtxGetDevice(&dev);
    if (r != CUDA_SUCCESS) return -1;

    cuDeviceGetAttribute(&out->max_threads_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    cuDeviceGetAttribute(&out->warp_size,
        CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);
    cuDeviceGetAttribute(&out->num_sms,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&out->max_regs_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, dev);
    cuDeviceGetAttribute(&out->max_smem_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, dev);
    cuDeviceGetAttribute(&out->max_blocks_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR, dev);
    return 0;
}

// Load PTX, query the kernel's resource usage, and compute theoretical
// occupancy.  The temporary CUmodule is unloaded before returning.
int occupancy_theoretical(const char *ptx, const char *func_name,
                          int block_size, size_t dyn_smem,
                          KernelOccupancy *out) {
    CUmodule mod;
    CUresult r = cuModuleLoadData(&mod, ptx);
    if (r != CUDA_SUCCESS) return -1;

    CUfunction func;
    r = cuModuleGetFunction(&func, mod, func_name);
    if (r != CUDA_SUCCESS) { cuModuleUnload(mod); return -2; }

    int regs = 0;
    cuFuncGetAttribute(&regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
    out->regs_per_thread = regs;

    int smem = 0;
    cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
    out->static_smem_bytes = smem;

    int max_blocks = 0;
    r = cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_blocks, func, block_size, dyn_smem);
    if (r != CUDA_SUCCESS) { cuModuleUnload(mod); return -3; }
    out->max_active_blocks_per_sm = max_blocks;

    // Compute theoretical occupancy as active_warps / max_warps.
    CUdevice dev;
    cuCtxGetDevice(&dev);
    int max_threads_per_sm = 0;
    cuDeviceGetAttribute(&max_threads_per_sm,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, dev);
    int warp_size = 32;
    cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);

    int max_warps_per_sm = max_threads_per_sm / warp_size;
    int warps_per_block  = (block_size + warp_size - 1) / warp_size;
    int active_warps     = max_blocks * warps_per_block;

    out->theoretical_occupancy = (max_warps_per_sm > 0)
        ? (float)active_warps / (float)max_warps_per_sm
        : 0.0f;

    cuModuleUnload(mod);
    return 0;
}
