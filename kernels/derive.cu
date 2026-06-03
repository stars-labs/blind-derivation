// BIP32 non-hardened public-side child derivation — Stage C.
//
// child_pubkey(i) = parent_pub + IL(i)*G, where
//   I       = HMAC-SHA512(chain_code, parent_pub || index_be)
//   IL,IR   = I[0..32], I[32..64]   (IL is a big-endian 256-bit integer)
// then hash160(child_pubkey). This is the public-key side (a point add) that
// no surveyed repo implements end-to-end; primitives come from the shared
// headers.
//
// The parent point is constant across the batch, so the host decompresses it
// once and passes affine (X,Y); the kernel does not do modular sqrt.

#include "secp256k1.cuh"
#include "hash.cuh"

// secp256k1 group order n, big-endian.
__device__ __constant__ uint8_t SECP_N_BE[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41};

// fe_from_be lives in secp256k1.cuh (shared with the fixed-base table).

// 1 if the 32 big-endian bytes are all zero.
__device__ int be_is_zero(const uint8_t *b) {
    uint8_t x = 0;
    for (int i = 0; i < 32; i++) x |= b[i];
    return x == 0;
}

// 1 if a >= b (both 32 big-endian bytes).
__device__ int be_ge(const uint8_t *a, const uint8_t *b) {
    for (int i = 0; i < 32; i++) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return 0;
    }
    return 1;
}

extern "C" __global__ void derive_child_kernel(
    const uint8_t *parent_pub33, const uint8_t *chain_code32,
    const uint8_t *parent_x32, const uint8_t *parent_y32, const uint8_t *gtable,
    uint32_t start_index, uint32_t n,
    uint8_t *out_pub33, uint8_t *out_h160, uint8_t *out_status) {

    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t index = start_index + i;

    uint8_t *opub = out_pub33 + (size_t)i * 33;
    uint8_t *oh = out_h160 + (size_t)i * 20;

    // msg = parent_pub(33) || index_be(4)
    uint8_t msg[37];
    for (int j = 0; j < 33; j++) msg[j] = parent_pub33[j];
    msg[33] = (uint8_t)(index >> 24);
    msg[34] = (uint8_t)(index >> 16);
    msg[35] = (uint8_t)(index >> 8);
    msg[36] = (uint8_t)(index);

    uint8_t I[64];
    hmac_sha512(chain_code32, 32, msg, 37, I);
    const uint8_t *IL = I; // big-endian 256-bit integer

    // BIP32 validity: IL must be in [1, n-1].
    if (be_is_zero(IL) || be_ge(IL, SECP_N_BE)) {
        out_status[i] = 1;
        for (int j = 0; j < 33; j++) opub[j] = 0;
        for (int j = 0; j < 20; j++) oh[j] = 0;
        return;
    }

    // IL*G via fixed-base table
    jpoint ilg;
    scalar_mul_g_table(IL, gtable, &ilg);

    // parent point (affine, decompressed on host) as Jacobian Z=1
    jpoint parent;
    fe_from_be(parent.X, parent_x32);
    fe_from_be(parent.Y, parent_y32);
    uint32_t one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    fe_set(parent.Z, one);

    jpoint child;
    jp_add(&child, &ilg, &parent);

    if (jp_is_inf(&child)) {
        out_status[i] = 1;
        for (int j = 0; j < 33; j++) opub[j] = 0;
        for (int j = 0; j < 20; j++) oh[j] = 0;
        return;
    }

    jp_compress(&child, opub);

    // hash160(child_pubkey)
    uint8_t digest[32];
    sha256_oneblock(opub, 33, digest);
    ripemd160_oneblock(digest, 32, oh);

    out_status[i] = 0;
}
