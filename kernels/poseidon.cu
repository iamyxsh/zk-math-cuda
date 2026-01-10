#include <stdint.h>

#define T 3
#define RF 8
#define RP 56
#define TOTAL_ROUNDS (RF + RP)
#define N_LIMBS 8

// BLS12-381 scalar field modulus in 32-bit limbs (little-endian)
__device__ __constant__ uint32_t MODULUS[N_LIMBS] = {
    0x00000001, 0xffffffff, 0xfffe5bfe, 0x53bda402,
    0x09a1d805, 0x3339d808, 0x299d7d48, 0x73eda753
};

// -p^(-1) mod 2^32 = 0xffffffff (since p[0] = 1, inv(1) = 1, neg = -1)
__device__ __constant__ uint32_t INV32 = 0xffffffff;

// R^2 mod p in 32-bit limbs (for converting to Montgomery form)
__device__ __constant__ uint32_t R2[N_LIMBS] = {
    0xf3f29c6d, 0xc999e990, 0x87925c23, 0x2b6cedcb,
    0x7254398f, 0x05d31496, 0x9f59ff11, 0x0748d9d9
};

typedef struct { uint32_t limbs[N_LIMBS]; } FpGpu;

// --- Field arithmetic (32-bit limbs, Montgomery form) ---

__device__ int fp_gte(const uint32_t a[N_LIMBS], const uint32_t b[N_LIMBS]) {
    for (int i = N_LIMBS - 1; i >= 0; i--) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;
}

__device__ void fp_sub_assign(uint32_t a[N_LIMBS], const uint32_t b[N_LIMBS]) {
    uint32_t borrow = 0;
    for (int j = 0; j < N_LIMBS; j++) {
        uint64_t diff = (uint64_t)a[j] - b[j] - borrow;
        a[j] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
}

__device__ void fp_add(FpGpu* result, const FpGpu* a, const FpGpu* b) {
    uint32_t carry = 0;
    for (int j = 0; j < N_LIMBS; j++) {
        uint64_t s = (uint64_t)a->limbs[j] + b->limbs[j] + carry;
        result->limbs[j] = (uint32_t)s;
        carry = (uint32_t)(s >> 32);
    }
    if (carry || fp_gte(result->limbs, MODULUS)) {
        fp_sub_assign(result->limbs, MODULUS);
    }
}

// CIOS Montgomery multiplication
__device__ void fp_mul(FpGpu* result, const FpGpu* a, const FpGpu* b) {
    uint32_t t[N_LIMBS + 1] = {0};

    for (int i = 0; i < N_LIMBS; i++) {
        uint32_t c = 0;
        for (int j = 0; j < N_LIMBS; j++) {
            uint64_t x = (uint64_t)t[j] + (uint64_t)a->limbs[j] * b->limbs[i] + c;
            t[j] = (uint32_t)x;
            c = (uint32_t)(x >> 32);
        }
        uint64_t x = (uint64_t)t[N_LIMBS] + c;
        t[N_LIMBS] = (uint32_t)x;

        uint32_t m = t[0] * INV32;
        x = (uint64_t)t[0] + (uint64_t)m * MODULUS[0];
        c = (uint32_t)(x >> 32);

        for (int j = 1; j < N_LIMBS; j++) {
            x = (uint64_t)t[j] + (uint64_t)m * MODULUS[j] + c;
            t[j - 1] = (uint32_t)x;
            c = (uint32_t)(x >> 32);
        }
        x = (uint64_t)t[N_LIMBS] + c;
        t[N_LIMBS - 1] = (uint32_t)x;
        t[N_LIMBS] = (uint32_t)(x >> 32);
    }

    // Conditional subtraction
    uint32_t tmp[N_LIMBS];
    uint32_t borrow = 0;
    for (int j = 0; j < N_LIMBS; j++) {
        uint64_t diff = (uint64_t)t[j] - MODULUS[j] - borrow;
        tmp[j] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }
    int subtract = (t[N_LIMBS] >= borrow);
    for (int j = 0; j < N_LIMBS; j++) {
        result->limbs[j] = subtract ? tmp[j] : t[j];
    }
}

__device__ void fp_pow5(FpGpu* result, const FpGpu* a) {
    FpGpu x2, x4;
    fp_mul(&x2, a, a);
    fp_mul(&x4, &x2, &x2);
    fp_mul(result, &x4, a);
}

// Convert a small integer to Montgomery form on device
__device__ void fp_from_u32(FpGpu* result, uint32_t val) {
    FpGpu raw = {{0}};
    raw.limbs[0] = val;
    FpGpu r2;
    for (int j = 0; j < N_LIMBS; j++) r2.limbs[j] = R2[j];
    fp_mul(result, &raw, &r2);
}

// --- Poseidon permutation kernel ---

extern "C" __global__ void poseidon_permutation_kernel(
    const FpGpu* __restrict__ input,
    FpGpu* __restrict__ output,
    uint32_t batch_size
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    FpGpu state[T];
    for (int j = 0; j < T; j++) {
        state[j] = input[tid * T + j];
    }

    // MDS matrix: [[2,1,1],[1,2,1],[1,1,2]]
    FpGpu mds_one, mds_two;
    fp_from_u32(&mds_one, 1);
    fp_from_u32(&mds_two, 2);

    for (int r = 0; r < TOTAL_ROUNDS; r++) {
        // AddRoundConstants: rc[i] = from_u32(r * T + j + 1)
        for (int j = 0; j < T; j++) {
            FpGpu rc;
            fp_from_u32(&rc, (uint32_t)(r * T + j + 1));
            fp_add(&state[j], &state[j], &rc);
        }

        // SubWords (S-box: x^5)
        if (r < RF / 2 || r >= RF / 2 + RP) {
            for (int j = 0; j < T; j++) {
                FpGpu tmp;
                fp_pow5(&tmp, &state[j]);
                state[j] = tmp;
            }
        } else {
            FpGpu tmp;
            fp_pow5(&tmp, &state[0]);
            state[0] = tmp;
        }

        // MixLayer: MDS * state
        FpGpu new_state[T];
        for (int i = 0; i < T; i++) {
            FpGpu acc = {{0}};
            for (int j = 0; j < T; j++) {
                FpGpu term;
                FpGpu* coeff = (i == j) ? &mds_two : &mds_one;
                fp_mul(&term, coeff, &state[j]);
                fp_add(&acc, &acc, &term);
            }
            new_state[i] = acc;
        }
        for (int j = 0; j < T; j++) {
            state[j] = new_state[j];
        }
    }

    for (int j = 0; j < T; j++) {
        output[tid * T + j] = state[j];
    }
}
