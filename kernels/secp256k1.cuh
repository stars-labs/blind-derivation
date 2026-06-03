#pragma once
// secp256k1 field + group arithmetic on GPU — Stage B.
//
// Correctness-first implementation: 8x32-bit-limb field arithmetic mod
// p = 2^256 - 2^32 - 977, Jacobian point add/double, and a plain
// double-and-add scalar multiply of the generator (IL*G). The windowed
// fixed-base table (CudaBrainSecp style) is a Stage E optimization; this file
// is the verifiable baseline, validated byte-for-byte against the k256 crate.
//
// Limb layout: uint32_t[8], little-endian (limb[0] is least significant).

#include <stdint.h>

typedef uint32_t fe[8];

// p = 2^256 - 2^32 - 977
__device__ __constant__ uint32_t FE_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};

// Generator G (affine), little-endian limbs.
__device__ __constant__ uint32_t FE_GX[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E};
__device__ __constant__ uint32_t FE_GY[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77};

// Exponent p-2 for Fermat inverse, big-endian bytes.
__device__ __constant__ uint8_t FE_P_MINUS_2[32] = {
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFE, 0xFF, 0xFF, 0xFC, 0x2D};

__device__ void fe_set(fe r, const uint32_t *a) {
    for (int i = 0; i < 8; i++) r[i] = a[i];
}
__device__ void fe_set_zero(fe r) {
    for (int i = 0; i < 8; i++) r[i] = 0;
}
__device__ int fe_is_zero(const fe a) {
    uint32_t x = 0;
    for (int i = 0; i < 8; i++) x |= a[i];
    return x == 0;
}
__device__ int fe_eq(const fe a, const fe b) {
    uint32_t x = 0;
    for (int i = 0; i < 8; i++) x |= (a[i] ^ b[i]);
    return x == 0;
}
// Returns 1 if a >= p.
__device__ int fe_ge_p(const fe a) {
    for (int i = 7; i >= 0; i--) {
        if (a[i] > FE_P[i]) return 1;
        if (a[i] < FE_P[i]) return 0;
    }
    return 1; // equal
}
__device__ void fe_sub_p(fe a) {
    uint64_t borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t cur = (uint64_t)a[i] - FE_P[i] - borrow;
        a[i] = (uint32_t)cur;
        borrow = (cur >> 63) & 1;
    }
}

__device__ void fe_add(fe r, const fe a, const fe b) {
    uint64_t c = 0;
    for (int i = 0; i < 8; i++) {
        c += (uint64_t)a[i] + b[i];
        r[i] = (uint32_t)c;
        c >>= 32;
    }
    // Inputs are < p, so the sum is < 2p and c is 0 or 1.
    // Fold the carry: 2^256 == 2^32 + 977 (mod p), i.e. add 0x1000003D1.
    if (c) {
        uint64_t cc = (uint64_t)r[0] + 977u;
        r[0] = (uint32_t)cc;
        cc >>= 32;
        cc += (uint64_t)r[1] + 1u;
        r[1] = (uint32_t)cc;
        cc >>= 32;
        for (int i = 2; i < 8 && cc; i++) {
            cc += r[i];
            r[i] = (uint32_t)cc;
            cc >>= 32;
        }
    }
    if (fe_ge_p(r)) fe_sub_p(r);
}

__device__ void fe_sub(fe r, const fe a, const fe b) {
    int64_t borrow = 0;
    uint32_t tmp[8];
    for (int i = 0; i < 8; i++) {
        int64_t cur = (int64_t)a[i] - b[i] - borrow;
        tmp[i] = (uint32_t)cur;
        borrow = (cur < 0) ? 1 : 0;
    }
    if (borrow) {
        // add p back
        uint64_t c = 0;
        for (int i = 0; i < 8; i++) {
            c += (uint64_t)tmp[i] + FE_P[i];
            tmp[i] = (uint32_t)c;
            c >>= 32;
        }
    }
    for (int i = 0; i < 8; i++) r[i] = tmp[i];
}

// Reduce a 16-limb value (little-endian) modulo p into r[8].
__device__ void fe_reduce_wide(uint32_t t[16], fe r) {
    // Repeatedly fold limbs 8.. into 0.. using 2^256 == 2^32 + 977.
    for (int pass = 0; pass < 5; pass++) {
        uint32_t h[8];
        uint32_t any = 0;
        for (int i = 0; i < 8; i++) {
            h[i] = t[8 + i];
            any |= h[i];
            t[8 + i] = 0;
        }
        if (!any) break;
        // t[0..] += h * 977
        uint64_t carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t cur = (uint64_t)t[i] + (uint64_t)h[i] * 977u + carry;
            t[i] = (uint32_t)cur;
            carry = cur >> 32;
        }
        for (int i = 8; carry; i++) {
            uint64_t cur = (uint64_t)t[i] + carry;
            t[i] = (uint32_t)cur;
            carry = cur >> 32;
        }
        // t[1..] += h (i.e. h << 32 bits)
        carry = 0;
        for (int i = 0; i < 8; i++) {
            uint64_t cur = (uint64_t)t[i + 1] + h[i] + carry;
            t[i + 1] = (uint32_t)cur;
            carry = cur >> 32;
        }
        for (int i = 9; carry; i++) {
            uint64_t cur = (uint64_t)t[i] + carry;
            t[i] = (uint32_t)cur;
            carry = cur >> 32;
        }
    }
    for (int i = 0; i < 8; i++) r[i] = t[i];
    if (fe_ge_p(r)) fe_sub_p(r);
    if (fe_ge_p(r)) fe_sub_p(r);
}

__device__ void fe_mul(fe r, const fe a, const fe b) {
    uint32_t t[16];
    for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t cur = (uint64_t)a[i] * b[j] + t[i + j] + carry;
            t[i + j] = (uint32_t)cur;
            carry = cur >> 32;
        }
        t[i + 8] += (uint32_t)carry;
    }
    fe_reduce_wide(t, r);
}

__device__ void fe_sqr(fe r, const fe a) { fe_mul(r, a, a); }

// r = a^(p-2) mod p  (modular inverse via Fermat).
__device__ void fe_inv(fe r, const fe a) {
    fe result;
    uint32_t one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    fe_set(result, one);
    for (int byte = 0; byte < 32; byte++) {
        uint8_t e = FE_P_MINUS_2[byte];
        for (int bit = 7; bit >= 0; bit--) {
            fe_sqr(result, result);
            if ((e >> bit) & 1) fe_mul(result, result, a);
        }
    }
    fe_set(r, result);
}

// ---------------------------------------------------------------------------
// Jacobian point arithmetic (a = 0). Infinity is represented by Z == 0.
// ---------------------------------------------------------------------------

typedef struct {
    fe X, Y, Z;
} jpoint;

__device__ int jp_is_inf(const jpoint *p) { return fe_is_zero(p->Z); }

__device__ void jp_set_inf(jpoint *p) {
    fe_set_zero(p->X);
    fe_set_zero(p->Y);
    fe_set_zero(p->Z);
}

// dbl-2009-l (a = 0).
__device__ void jp_double(jpoint *r, const jpoint *p) {
    if (fe_is_zero(p->Z) || fe_is_zero(p->Y)) {
        jp_set_inf(r);
        return;
    }
    fe XX, YY, YYYY, ZZ, S, M, T, t0, t1;
    fe_sqr(XX, p->X);
    fe_sqr(YY, p->Y);
    fe_sqr(YYYY, YY);
    fe_sqr(ZZ, p->Z);
    // S = 2*((X+YY)^2 - XX - YYYY)
    fe_add(t0, p->X, YY);
    fe_sqr(t0, t0);
    fe_sub(t0, t0, XX);
    fe_sub(t0, t0, YYYY);
    fe_add(S, t0, t0);
    // M = 3*XX
    fe_add(M, XX, XX);
    fe_add(M, M, XX);
    // T = M^2 - 2*S
    fe_sqr(T, M);
    fe_sub(T, T, S);
    fe_sub(T, T, S);
    fe_set(r->X, T);
    // Y3 = M*(S - T) - 8*YYYY
    fe_sub(t0, S, T);
    fe_mul(t0, M, t0);
    fe_add(t1, YYYY, YYYY);
    fe_add(t1, t1, t1);
    fe_add(t1, t1, t1); // 8*YYYY
    fe_sub(r->Y, t0, t1);
    // Z3 = (Y+Z)^2 - YY - ZZ
    fe_add(t0, p->Y, p->Z);
    fe_sqr(t0, t0);
    fe_sub(t0, t0, YY);
    fe_sub(t0, t0, ZZ);
    fe_set(r->Z, t0);
}

// add-2007-bl (general Jacobian + Jacobian).
__device__ void jp_add(jpoint *r, const jpoint *p, const jpoint *q) {
    if (jp_is_inf(p)) { *r = *q; return; }
    if (jp_is_inf(q)) { *r = *p; return; }
    fe Z1Z1, Z2Z2, U1, U2, S1, S2, t0;
    fe_sqr(Z1Z1, p->Z);
    fe_sqr(Z2Z2, q->Z);
    fe_mul(U1, p->X, Z2Z2);
    fe_mul(U2, q->X, Z1Z1);
    fe_mul(S1, p->Y, q->Z);
    fe_mul(S1, S1, Z2Z2);
    fe_mul(S2, q->Y, p->Z);
    fe_mul(S2, S2, Z1Z1);
    if (fe_eq(U1, U2)) {
        if (!fe_eq(S1, S2)) { jp_set_inf(r); return; }
        jp_double(r, p);
        return;
    }
    fe H, I, J, R, V;
    fe_sub(H, U2, U1);
    fe_add(I, H, H);
    fe_sqr(I, I);       // (2H)^2
    fe_mul(J, H, I);
    fe_sub(R, S2, S1);
    fe_add(R, R, R);    // 2*(S2-S1)
    fe_mul(V, U1, I);
    // X3 = R^2 - J - 2V
    fe_sqr(r->X, R);
    fe_sub(r->X, r->X, J);
    fe_sub(r->X, r->X, V);
    fe_sub(r->X, r->X, V);
    // Y3 = R*(V - X3) - 2*S1*J
    fe_sub(t0, V, r->X);
    fe_mul(t0, R, t0);
    fe_mul(S1, S1, J);
    fe_add(S1, S1, S1);
    fe_sub(r->Y, t0, S1);
    // Z3 = ((Z1+Z2)^2 - Z1Z1 - Z2Z2) * H
    fe_add(t0, p->Z, q->Z);
    fe_sqr(t0, t0);
    fe_sub(t0, t0, Z1Z1);
    fe_sub(t0, t0, Z2Z2);
    fe_mul(r->Z, t0, H);
}

// R = k*G via MSB-first double-and-add. k is 32 big-endian bytes.
__device__ void scalar_mul_g(const uint8_t *k_be, jpoint *out) {
    jpoint R;
    jp_set_inf(&R);
    jpoint G;
    fe_set(G.X, FE_GX);
    fe_set(G.Y, FE_GY);
    uint32_t one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    fe_set(G.Z, one);
    for (int i = 255; i >= 0; i--) {
        jpoint tmp;
        jp_double(&tmp, &R);
        R = tmp;
        int byte = 31 - (i >> 3);
        int bit = i & 7;
        if ((k_be[byte] >> bit) & 1) {
            jpoint tmp2;
            jp_add(&tmp2, &R, &G);
            R = tmp2;
        }
    }
    *out = R;
}

// Convert Jacobian point to a 33-byte compressed pubkey (big-endian X).
__device__ void jp_compress(const jpoint *p, uint8_t *out33) {
    if (jp_is_inf(p)) {
        for (int i = 0; i < 33; i++) out33[i] = 0;
        return;
    }
    fe zinv, zinv2, zinv3, x, y;
    fe_inv(zinv, p->Z);
    fe_sqr(zinv2, zinv);
    fe_mul(zinv3, zinv2, zinv);
    fe_mul(x, p->X, zinv2);
    fe_mul(y, p->Y, zinv3);
    out33[0] = 0x02 | (uint8_t)(y[0] & 1);
    for (int i = 0; i < 8; i++) {
        uint32_t w = x[7 - i];
        out33[1 + i * 4] = (uint8_t)(w >> 24);
        out33[1 + i * 4 + 1] = (uint8_t)(w >> 16);
        out33[1 + i * 4 + 2] = (uint8_t)(w >> 8);
        out33[1 + i * 4 + 3] = (uint8_t)(w);
    }
}


// ---------------------------------------------------------------------------
// Fixed-base scalar multiply via a precomputed windowed table (Stage E).
//
// table layout: 32 windows x 255 entries, each entry an affine point stored as
// 64 big-endian bytes (X || Y). entry (w, d) = d * 256^w * G, for d in 1..255.
// Then k*G = sum over windows w of digit_w * 256^w * G, where digit_w is the
// w-th base-256 digit of k. 32 point additions, no doublings.
// ---------------------------------------------------------------------------

// Load a field element from 32 big-endian bytes into little-endian limbs.
__device__ void fe_from_be(fe r, const uint8_t *b) {
    for (int i = 0; i < 8; i++) {
        int off = (7 - i) * 4;
        r[i] = ((uint32_t)b[off] << 24) | ((uint32_t)b[off + 1] << 16) |
               ((uint32_t)b[off + 2] << 8) | (uint32_t)b[off + 3];
    }
}

__device__ void scalar_mul_g_table(const uint8_t *k_be, const uint8_t *table,
                                   jpoint *out) {
    jpoint R;
    jp_set_inf(&R);
    uint32_t one[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    for (int w = 0; w < 32; w++) {
        uint8_t digit = k_be[31 - w]; // window w == base-256 digit w
        if (digit == 0) continue;
        const uint8_t *e = table + ((size_t)w * 255 + (digit - 1)) * 64;
        jpoint P;
        fe_from_be(P.X, e);
        fe_from_be(P.Y, e + 32);
        fe_set(P.Z, one);
        jpoint t;
        jp_add(&t, &R, &P);
        R = t;
    }
    *out = R;
}
