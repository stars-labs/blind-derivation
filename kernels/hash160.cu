// SHA-256 / HASH160 test kernels — device code lives in hash.cuh.
#include "hash.cuh"

extern "C" __global__ void sha256_kernel(const uint8_t *msgs, const uint32_t *lens,
                                         uint32_t stride, uint32_t n, uint8_t *out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    sha256_oneblock(msgs + (size_t)i * stride, lens[i], out + (size_t)i * 32);
}

extern "C" __global__ void hash160_kernel(const uint8_t *msgs, const uint32_t *lens,
                                          uint32_t stride, uint32_t n, uint8_t *out) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t digest[32];
    sha256_oneblock(msgs + (size_t)i * stride, lens[i], digest);
    ripemd160_oneblock(digest, 32, out + (size_t)i * 20);
}
