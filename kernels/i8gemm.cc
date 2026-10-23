//===- i8gemm.cc ----------------------------------------------- C++ -----===//
// Q8_0 int8 GEMM (M tokens x GW out, GK k) — Qwen3EmbeddingXDNA.
// aie::mmul<8,8,8,int8,acc32>: tile = 8m x 8n x 8k. One MMUL accumulates a
// full Q8_0 k-block (32 k = 4 tiles, one mul + 3 mac); vector epilogue
// (f32 cvt + fma vs dvec) folds into f32 gf[64]; row-group end -> out.
// Stream objs (OB), TPO weight tiles/obj (block-aligned):
//   [0] meta f32 act_scale | [1..CX] A [kt][m*8+i] | [.. NSD] d16 f16
//   [.. NWT] W tiles [rg][kt][i*8+n]. Drain: MM*GW f32.
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef GW
#define GW 512
#endif
#ifndef GK
#define GK 512
#endif
#ifndef MM
#define MM 8
#endif
#ifndef OB
#define OB 3456
#endif

using MMUL = ::aie::mmul<8, 8, 8, int8_t, int8_t, acc32>;

namespace {
constexpr int KT = GK / 8;                  // k-tiles per row group
constexpr int KB = GK / 32;                 // q8 blocks per row
constexpr int RG = GW / 8;
constexpr int TILES = RG * KT;
constexpr int TPO = (OB / 64) / 4 * 4;      // tiles/obj, block-aligned (52)
constexpr int NWT = (TILES + TPO - 1) / TPO;
constexpr int ABYTES = KT * 64;
constexpr int CX = (ABYTES + OB - 1) / OB;
constexpr int NDB = GW * KB * 2;            // d16 bytes
constexpr int NSD = (NDB + OB - 1) / OB;
constexpr int NOD = (MM * GW * 4 + OB - 1) / OB;
}

static float g_as;
static int g_rg0;
static float g_gf[64];
static int g_cur;

// P layout: [0,ABYTES) A | [DOFF,DOFF+NDB) d16 | [POFF,..) f32 out
constexpr int DOFF = (ABYTES + 4095) & ~4095;
constexpr int POFF = (DOFF + NDB + 4095) & ~4095;

static inline float f16f(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    float v;
    if (e == 0) v = (float)m * 0x1p-24f;
    else if (e == 0x1F) v = m ? 0.f : (h & 0x8000 ? -1.f / 0.f : 1.f / 0.f);
    else {
        uint32_t b = ((h & 0x8000u) << 16) | ((e + 112u) << 23) | (m << 13);
        __builtin_memcpy(&v, &b, 4);
    }
    return v;
}

extern "C" {

__attribute__((noinline)) void i8_reset() { g_cur = 0; }

__attribute__((noinline)) void i8_step(const uint8_t *__restrict obj,
                                       uint8_t *__restrict pbuf) {
    const int o = g_cur++;
    if (o == 0) { __builtin_memcpy(&g_as, obj, 4);
                     __builtin_memcpy(&g_rg0, obj + 4, 4); return; }
    if (o <= CX) {
        const int n = ABYTES - (o - 1) * OB < OB ? ABYTES - (o - 1) * OB : OB;
        __builtin_memcpy(pbuf + (o - 1) * OB, obj, n);
        return;
    }
    if (o <= CX + NSD) {
        const int n = NDB - (o - CX - 1) * OB < OB
                          ? NDB - (o - CX - 1) * OB : OB;
        __builtin_memcpy(pbuf + DOFF + (o - CX - 1) * OB, obj, n);
        return;
    }
    const int64_t a0 = (int64_t)o - CX - NSD - 1;
    const int t0 = (int)(a0 * TPO);
    const int nt = TILES - t0 < TPO ? TILES - t0 : TPO;
    const uint16_t *dall = reinterpret_cast<const uint16_t *>(pbuf + DOFF);
    const int8_t *A = reinterpret_cast<const int8_t *>(pbuf);
    for (int t = t0; t < t0 + nt; t += 4) {            // one k-block
        const int rg = g_rg0 + t / KT, kt = t % KT;
        if (kt == 0)
            for (int i = 0; i < 64; i++) g_gf[i] = 0.f;
        MMUL c;
        c.mul(::aie::load_v<64>(A + (kt & ~3) * 64),
              ::aie::load_v<64>(reinterpret_cast<const int8_t *>(obj) +
                                (t - t0) * 64));
        for (int i = 1; i < 4; i++)
            c.mac(::aie::load_v<64>(A + ((kt & ~3) + i) * 64),
                  ::aie::load_v<64>(reinterpret_cast<const int8_t *>(obj) +
                                    (t - t0 + i) * 64));
        // TIMING VARIANT: drain accum only every 4th k-block (numerics
        // wrong by design; measures per-k-block accum-read cost).
        if (((kt >> 2) & 3) == 0) {
            ::aie::accum<acc32, 64> ac = c.to_accum();
            const uint16_t *dp = dall + (rg - g_rg0) * 8 * KB + (kt >> 2);
            float dd[8];
            for (int q = 0; q < 8; q++) dd[q] = f16f(dp[q * KB]);
            auto cv0 = ac.extract<32>(0);
            auto cv1 = ac.extract<32>(1);
            const ::aie::vector<int32_t, 32> iv0 = cv0.to_vector<int32_t>();
            const ::aie::vector<int32_t, 32> iv1 = cv1.to_vector<int32_t>();
            const int32_t *p0 = (const int32_t *)&iv0, *p1 = (const int32_t *)&iv1;
            for (int l = 0; l < 32; l++) {
                g_gf[l] += (float)p0[l] * dd[l & 7];
                g_gf[32 + l] += (float)p1[l] * dd[l & 7];
            }
        } else {
            float *z = reinterpret_cast<float *>(pbuf + POFF);
            if (obj == reinterpret_cast<const uint8_t *>(z)) g_gf[0] = 1.f;
        }
        if (kt == KT - 4) {                          // last k-block of row group
            float *out = reinterpret_cast<float *>(pbuf + POFF);
            for (int m = 0; m < MM; m++)
                for (int n = 0; n < 8; n++)
                    out[m * GW + (rg - g_rg0) * 8 + n] = g_gf[m * 8 + n] * g_as;
        }
    }
}

__attribute__((noinline)) void i8_flush(uint8_t *__restrict obj,
                                        const uint8_t *__restrict pbuf) {
    static int g_od = 0;
    const int total = MM * GW * 4;
    const int off = g_od * OB;
    const int n = total - off < OB ? total - off : OB;
    __builtin_memcpy(obj, pbuf + POFF + off, n);
    if (++g_od >= NOD) g_od = 0;
}

// Indexed aliases: 8 shim workers each need a distinct Kernel symbol.
#define I8_WRAP(i)                                              \
    void i8_reset##i(void) { g_cur = 0; }                       \
    void i8_step##i(const uint8_t *o, uint8_t *p) { i8_step(o, p); } \
    void i8_flush##i(uint8_t *o, const uint8_t *p) { i8_flush(o, p); }
#define I8_ALL(i) I8_WRAP(i)
I8_ALL(0) I8_ALL(1) I8_ALL(2) I8_ALL(3)
I8_ALL(4) I8_ALL(5) I8_ALL(6) I8_ALL(7)

}
