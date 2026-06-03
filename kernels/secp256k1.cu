// secp256k1 k*G test kernel — device code lives in secp256k1.cuh.
#include "secp256k1.cuh"

// Kernel: one thread per scalar. scalars: n*32 (BE). out: n*33 compressed.
// Uses the fixed-base windowed table (validates the Stage E path vs k256).
extern "C" __global__ void scalar_mul_g_kernel(const uint8_t *scalars,
                                              const uint8_t *gtable, uint32_t n,
                                              uint8_t *out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    jpoint R;
    scalar_mul_g_table(scalars + (size_t)i * 32, gtable, &R);
    jp_compress(&R, out + (size_t)i * 33);
}
