
# Project Title

A brief description of what this project does and who it's for

# zk-profiler

GPU kernel profiler for zero-knowledge proof primitives, built with Rust + CUDA.

Implements a **Poseidon hash permutation** over the BLS12-381 scalar field on GPU, then profiles it using NVIDIA's CUPTI instrumentation APIs — measuring kernel timing, warp stall breakdowns, and occupancy.

---

## Architecture

```
                   ┌──────────────────────────────┐
                   │         zk-profiler          │
                   │        (Rust binary)         │
                   └──────┬───────────────┬───────┘
                          │               │
                ┌─────────┘               └──────────┐
                ▼                                    ▼
    ┌──────────────────────────┐           ┌──────────────────────────┐
    │    Compute Pipeline      │           │    Profiling Pipeline    │
    │                          │           │                          │
    │  field.rs    BLS12-381   │           │  cupti.rs     Activity   │
    │              Fp arith    │           │               API timing │
    │  poseidon.rs CPU ref     │           │  stall_       Event API  │
    │  poseidon.cu GPU kernel  │           │  counters.rs  warp stall │
    │  main.rs     launcher    │           │  occupancy.rs theo vs    │
    │                          │           │               achieved   │
    └──────────────────────────┘           └──────────────────────────┘
                │                                    │
                ▼                                    ▼
    ┌──────────────────────────┐           ┌──────────────────────────┐
    │   cudarc 0.12            │           │   C wrappers (cc crate)  │
    │   Rust CUDA driver       │           │                          │
    │   - PTX loading          │           │   - cupti_wrapper.c      │
    │   - htod/dtoh            │           │   - stall_counters.c     │
    │   - kernel launch        │           │   - occupancy.c          │
    └──────────────────────────┘           └──────────────────────────┘
                │                                    │
                └────────────────┬───────────────────┘
                                 ▼
                  ┌────────────────────────────┐
                  │    CUDA Runtime / Driver   │
                  │    + CUPTI (GPU hardware)  │
                  └────────────────────────────┘
```

## Data Flow

```
    Input states [Fp; 3]
           │
           ▼
    ┌──────────────┐    fp_to_gpu_limbs()   ┌──────────────┐
    │  CPU (Rust)  │ ─────────────────────▶ │  GPU (CUDA)  │
    │              │     htod_sync_copy     │              │
    │  [u64; 4]    │                        │  [u32; 8]    │
    │  Montgomery  │                        │  Montgomery  │
    └──────────────┘                        └──────┬───────┘
                                                   │
                                      poseidon_permutation_kernel
                                      ┌────────────┴───────────┐
                                      │  for each state:       │
                                      │  ┌──────────────────┐  │
                                      │  │ Full rounds 0-3  │  │
                                      │  │  S-box all elems │  │
                                      │  │  MDS multiply    │  │
                                      │  │  Add round const │  │
                                      │  ├──────────────────┤  │
                                      │  │ Partial 4-59     │  │
                                      │  │  S-box state[0]  │  │
                                      │  │  MDS multiply    │  │
                                      │  │  Add round const │  │
                                      │  ├──────────────────┤  │
                                      │  │ Full rounds60-63 │  │
                                      │  │  S-box all elems │  │
                                      │  │  MDS multiply    │  │
                                      │  │  Add round const │  │
                                      │  └──────────────────┘  │
                                      └────────────┬───────────┘
                                                   │
    ┌──────────────┐   fp_from_gpu_limbs()  ┌──────┴───────┐
    │  CPU (Rust)  │ ◀──────────────────────│  GPU result  │
    │              │     dtoh_sync_copy     │              │
    └──────────────┘                        └──────────────┘
```

## Profiling Stack

```
    ┌────────────────────┬────────────────────┬────────────────────┐
    │                    │  CUPTI Profiling   │                    │
    ├────────────────────┼────────────────────┼────────────────────┤
    │  Activity API      │  Event API         │  Driver API        │
    │  (cupti.rs)        │  (stall_counters)  │  (occupancy.rs)    │
    │                    │                    │                    │
    │  kernel start/end  │  stall_memory_     │  cuDeviceGet-      │
    │  timestamps        │  dependency        │  Attribute         │
    │                    │                    │                    │
    │  multi-launch      │  stall_exec_       │  cuOccupancyMax-   │
    │  variance          │  dependency        │  ActiveBlocks      │
    │                    │                    │                    │
    │  min/max/mean/     │  stall_inst_fetch  │  cuFuncGet-        │
    │  stddev            │                    │  Attribute         │
    │                    │  stall_sync        │                    │
    │                    │  + 4 more          │  theo vs achieved  │
    └────────────────────┴────────────────────┴────────────────────┘
```

## Field Arithmetic

BLS12-381 scalar field (`Fp`) with Montgomery multiplication:

```
    Modulus p = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

    Representation: [u64; 4] in Montgomery form  (CPU)
                    [u32; 8] in Montgomery form  (GPU)

    CPU (64-bit limbs)                 GPU (32-bit limbs)
    ┌─────────────────┐                ┌─────────────────┐
    │ CIOS mont_mul   │                │ CIOS mont_mul   │
    │ 4 outer iters   │                │ 8 outer iters   │
    │ 4 inner iters   │                │ 8 inner iters   │
    │ u128 products   │                │ u64 products    │
    └─────────────────┘                └─────────────────┘

    S-box: x^5 = x * x * x * x * x  (3 multiplications)
    MDS:   [[2,1,1],[1,2,1],[1,1,2]]
    Rounds: 8 full + 56 partial = 64 total
```

## Project Structure

```
zk-math-cuda/
├── Cargo.toml               # cudarc 0.12 + cc build dep
├── build.rs                  # compiles C wrappers, links CUPTI + CUDA
├── kernels/
│   ├── vector_add.ptx        # hand-written PTX (sm_52)
│   └── poseidon.cu           # CUDA C kernel (compile with nvcc)
└── src/
    ├── main.rs               # entry point, GPU launchers, tests
    ├── field.rs              # BLS12-381 Fp (Montgomery form)
    ├── poseidon.rs           # CPU reference Poseidon permutation
    ├── cupti.rs              # CUPTI Activity API — kernel timing
    ├── cupti_wrapper.c       # C shim for Activity API buffers
    ├── stall_counters.rs     # CUPTI Event API — warp stall counters
    ├── stall_counters.c      # C shim for Event API groups
    ├── occupancy.rs          # theoretical vs achieved occupancy
    └── occupancy.c           # C shim for cuOccupancy* + device props
```

## Build & Run

**Prerequisites:** CUDA toolkit (10+), `nvcc`, a CUDA-capable GPU.

```bash
# 1. Compile the Poseidon CUDA kernel to PTX
nvcc -ptx -arch=sm_XX kernels/poseidon.cu -o kernels/poseidon.ptx

# 2. Build and run
cargo run

# 3. Run tests
cargo test
```

Replace `sm_XX` with your GPU's compute capability (e.g. `sm_75` for Turing, `sm_86` for Ampere).

## Sample Output

```
GPU device 0 initialised
vector_add([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]) = [5.0, 7.0, 9.0]
poseidon GPU == CPU ✓

poseidon 20 launches  batch=256
  min     42.18 μs
  max     48.73 μs
  mean    43.91 μs
  stddev   1.67 μs

poseidon stall breakdown (batch=256):
  stall_memory_dependency             8234567
  stall_exec_dependency               4123456
  stall_inst_fetch                      12345
  stall_memory_throttle                 98765
  stall_sync                                0
  stall_texture                             0
  stall_other                            5432
  stall_not_selected                  1234567

occupancy (block=256, grid=1):
  SMs              40
  regs/thread      48
  max blocks/SM    4
  theoretical      50.0%
  achieved         12.5%
```

*(Values are illustrative; actual numbers depend on GPU model.)*

## Poseidon Parameters

| Parameter | Value |
|-----------|-------|
| Field | BLS12-381 scalar |
| Width (t) | 3 |
| Full rounds (RF) | 8 |
| Partial rounds (RP) | 56 |
| S-box | x^5 |
| MDS matrix | Cauchy-derived [[2,1,1],[1,2,1],[1,1,2]] |
| Round constants | Sequential integers 1..192 |

## Test Coverage

| Step | Tests | What |
|------|-------|------|
| 1 | 3 | GPU init, htod/dtoh roundtrip, zeroed alloc |
| 2 | 4 | vector_add basic, large, misaligned, empty |
| 3 | 9 | Fp arithmetic (in field.rs) |
| 4 | 5 | GPU vs CPU Poseidon, batch, empty, limb roundtrip |
| 5 | 3 | CUPTI timing, plausible duration, reset |
| 6 | 4 | multi-launch count, positive, stats invariants, batch scaling |
| 7 | 4 | stall profiler init, known events, memory_dep present, nonzero counts |
| 8 | 4 | device props, achieved <= theoretical, grid scaling, bounded |
| **Total** | **36** | |
