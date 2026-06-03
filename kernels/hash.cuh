#pragma once
// GPU hash kernels for blind-derivation — Stage A.
//
// Self-contained SHA-256, RIPEMD-160 and HASH160 = RIPEMD160(SHA256(x)).
// Single-block only: input messages must be <= 55 bytes, which covers every
// case this project needs (33-byte compressed pubkeys, the 3-byte "abc" test
// vector, and the 32-byte SHA-256 digest fed into RIPEMD-160).
//
// Validated on-host against known vectors:
//   SHA256("abc")   = ba7816bf...20015ad
//   RIPEMD160("abc")= 8eb208f7...15a0bfc
//   HASH160(0279BE66..F81798) = 751e76e8..433bd6   (matches address.rs:88)

#include <stdint.h>

// ---------------------------------------------------------------------------
// SHA-256 (big-endian), single 64-byte block.
// ---------------------------------------------------------------------------

__device__ __constant__ uint32_t SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

#define ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

// data: up to 55 bytes; out: 32 bytes (big-endian digest).
__device__ void sha256_oneblock(const uint8_t *data, uint32_t len, uint8_t *out) {
    uint8_t block[64];
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (uint32_t i = 0; i < len; i++) block[i] = data[i];
    block[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    block[56] = (uint8_t)(bits >> 56);
    block[57] = (uint8_t)(bits >> 48);
    block[58] = (uint8_t)(bits >> 40);
    block[59] = (uint8_t)(bits >> 32);
    block[60] = (uint8_t)(bits >> 24);
    block[61] = (uint8_t)(bits >> 16);
    block[62] = (uint8_t)(bits >> 8);
    block[63] = (uint8_t)(bits);

    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) | ((uint32_t)block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = ROTR32(w[i - 15], 7) ^ ROTR32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = ROTR32(w[i - 2], 17) ^ ROTR32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = 0x6a09e667, b = 0xbb67ae85, c = 0x3c6ef372, d = 0xa54ff53a;
    uint32_t e = 0x510e527f, f = 0x9b05688c, g = 0x1f83d9ab, h = 0x5be0cd19;

    for (int i = 0; i < 64; i++) {
        uint32_t S1 = ROTR32(e, 6) ^ ROTR32(e, 11) ^ ROTR32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = ROTR32(a, 2) ^ ROTR32(a, 13) ^ ROTR32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    uint32_t hs[8] = {0x6a09e667 + a, 0xbb67ae85 + b, 0x3c6ef372 + c, 0xa54ff53a + d,
                      0x510e527f + e, 0x9b05688c + f, 0x1f83d9ab + g, 0x5be0cd19 + h};
    for (int i = 0; i < 8; i++) {
        out[i * 4]     = (uint8_t)(hs[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(hs[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(hs[i] >> 8);
        out[i * 4 + 3] = (uint8_t)(hs[i]);
    }
}

// ---------------------------------------------------------------------------
// RIPEMD-160 (little-endian), single 64-byte block.
// ---------------------------------------------------------------------------

#define ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))
#define RM_F(x, y, z) ((x) ^ (y) ^ (z))
#define RM_G(x, y, z) (((x) & (y)) | ((~(x)) & (z)))
#define RM_H(x, y, z) (((x) | (~(y))) ^ (z))
#define RM_I(x, y, z) (((x) & (z)) | ((y) & (~(z))))
#define RM_J(x, y, z) ((x) ^ ((y) | (~(z))))

// data: up to 55 bytes; out: 20 bytes.
__device__ void ripemd160_oneblock(const uint8_t *data, uint32_t len, uint8_t *out) {
    uint8_t block[64];
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (uint32_t i = 0; i < len; i++) block[i] = data[i];
    block[len] = 0x80;
    uint64_t bits = (uint64_t)len * 8;
    block[56] = (uint8_t)(bits);
    block[57] = (uint8_t)(bits >> 8);
    block[58] = (uint8_t)(bits >> 16);
    block[59] = (uint8_t)(bits >> 24);
    block[60] = (uint8_t)(bits >> 32);
    block[61] = (uint8_t)(bits >> 40);
    block[62] = (uint8_t)(bits >> 48);
    block[63] = (uint8_t)(bits >> 56);

    uint32_t x[16];
    for (int i = 0; i < 16; i++) {
        x[i] = ((uint32_t)block[i * 4]) | ((uint32_t)block[i * 4 + 1] << 8) |
               ((uint32_t)block[i * 4 + 2] << 16) | ((uint32_t)block[i * 4 + 3] << 24);
    }

    // Message word order and rotation amounts (left and right lines).
    const int rl[80] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
                        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
                        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
                        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
                        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13};
    const int rr[80] = {5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
                        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
                        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
                        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
                        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11};
    const int sl[80] = {11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
                        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
                        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
                        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
                        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6};
    const int sr[80] = {8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
                        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
                        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
                        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
                        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11};
    const uint32_t kl[5] = {0x00000000,0x5a827999,0x6ed9eba1,0x8f1bbcdc,0xa953fd4e};
    const uint32_t kr[5] = {0x50a28be6,0x5c4dd124,0x6d703ef3,0x7a6d76e9,0x00000000};

    uint32_t al=0x67452301, bl=0xefcdab89, cl=0x98badcfe, dl=0x10325476, el=0xc3d2e1f0;
    uint32_t ar=al, br=bl, cr=cl, dr=dl, er=el;

    for (int j = 0; j < 80; j++) {
        int rnd = j / 16;
        uint32_t tl, tr;
        if (rnd == 0) { tl = RM_F(bl,cl,dl); tr = RM_J(br,cr,dr); }
        else if (rnd == 1) { tl = RM_G(bl,cl,dl); tr = RM_I(br,cr,dr); }
        else if (rnd == 2) { tl = RM_H(bl,cl,dl); tr = RM_H(br,cr,dr); }
        else if (rnd == 3) { tl = RM_I(bl,cl,dl); tr = RM_G(br,cr,dr); }
        else { tl = RM_J(bl,cl,dl); tr = RM_F(br,cr,dr); }

        tl = al + tl + x[rl[j]] + kl[rnd];
        tl = ROTL32(tl, sl[j]) + el;
        al = el; el = dl; dl = ROTL32(cl, 10); cl = bl; bl = tl;

        tr = ar + tr + x[rr[j]] + kr[rnd];
        tr = ROTL32(tr, sr[j]) + er;
        ar = er; er = dr; dr = ROTL32(cr, 10); cr = br; br = tr;
    }

    // Final combine (note the one-position rotation of the init constants).
    uint32_t res[5];
    res[0] = 0xefcdab89 + cl + dr;
    res[1] = 0x98badcfe + dl + er;
    res[2] = 0x10325476 + el + ar;
    res[3] = 0xc3d2e1f0 + al + br;
    res[4] = 0x67452301 + bl + cr;
    for (int i = 0; i < 5; i++) {
        out[i * 4]     = (uint8_t)(res[i]);
        out[i * 4 + 1] = (uint8_t)(res[i] >> 8);
        out[i * 4 + 2] = (uint8_t)(res[i] >> 16);
        out[i * 4 + 3] = (uint8_t)(res[i] >> 24);
    }
}


// ---------------------------------------------------------------------------
// SHA-512 (multi-block) and HMAC-SHA512.  Used by BIP32 child derivation.
// ---------------------------------------------------------------------------

__device__ __constant__ uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL};

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

__device__ void sha512_transform(uint64_t h[8], const uint8_t block[128]) {
    uint64_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = 0;
        for (int j = 0; j < 8; j++) w[i] = (w[i] << 8) | block[i * 8 + j];
    }
    for (int i = 16; i < 80; i++) {
        uint64_t s0 = ROTR64(w[i - 15], 1) ^ ROTR64(w[i - 15], 8) ^ (w[i - 15] >> 7);
        uint64_t s1 = ROTR64(w[i - 2], 19) ^ ROTR64(w[i - 2], 61) ^ (w[i - 2] >> 6);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint64_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 80; i++) {
        uint64_t S1 = ROTR64(e,14) ^ ROTR64(e,18) ^ ROTR64(e,41);
        uint64_t ch = (e & f) ^ ((~e) & g);
        uint64_t t1 = hh + S1 + ch + SHA512_K[i] + w[i];
        uint64_t S0 = ROTR64(a,28) ^ ROTR64(a,34) ^ ROTR64(a,39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t t2 = S0 + maj;
        hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e; h[5]+=f; h[6]+=g; h[7]+=hh;
}

// SHA-512 of a message up to ~250 bytes (buffer below bounds it). out: 64 bytes.
__device__ void sha512(const uint8_t *msg, uint32_t len, uint8_t out[64]) {
    uint64_t h[8] = {0x6a09e667f3bcc908ULL,0xbb67ae8584caa73bULL,0x3c6ef372fe94f82bULL,
                     0xa54ff53a5f1d36f1ULL,0x510e527fade682d1ULL,0x9b05688c2b3e6c1fULL,
                     0x1f83d9abfb41bd6bULL,0x5be0cd19137e2179ULL};
    uint8_t buf[384];
    for (uint32_t i = 0; i < len; i++) buf[i] = msg[i];
    buf[len] = 0x80;
    uint32_t padded = ((len + 1 + 16 + 127) / 128) * 128;
    for (uint32_t i = len + 1; i < padded; i++) buf[i] = 0;
    uint64_t bitlen = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) buf[padded - 1 - i] = (uint8_t)(bitlen >> (8 * i));
    for (uint32_t blk = 0; blk < padded / 128; blk++) sha512_transform(h, buf + blk * 128);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) out[i * 8 + j] = (uint8_t)(h[i] >> (56 - 8 * j));
}

// HMAC-SHA512 with key length <= 128 (our key is the 32-byte chain code).
// msg length kept small (<= 64); out: 64 bytes.
__device__ void hmac_sha512(const uint8_t *key, uint32_t keylen,
                            const uint8_t *msg, uint32_t msglen, uint8_t out[64]) {
    uint8_t ipad[128], opad[128];
    for (int i = 0; i < 128; i++) {
        uint8_t k = (i < (int)keylen) ? key[i] : 0;
        ipad[i] = k ^ 0x36;
        opad[i] = k ^ 0x5c;
    }
    uint8_t buf[256];
    for (int i = 0; i < 128; i++) buf[i] = ipad[i];
    for (uint32_t i = 0; i < msglen; i++) buf[128 + i] = msg[i];
    uint8_t inner[64];
    sha512(buf, 128 + msglen, inner);

    uint8_t buf2[192];
    for (int i = 0; i < 128; i++) buf2[i] = opad[i];
    for (int i = 0; i < 64; i++) buf2[128 + i] = inner[i];
    sha512(buf2, 128 + 64, out);
}
