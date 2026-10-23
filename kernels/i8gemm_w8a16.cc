//===- i8gemm_w8a16.cc ----------------------------------------- C++ -----===//
// W8A16 GEMM: 13-bit activations (hi/lo int8 pair) x per-channel int8 W.
// Long-K mac chain, 1 drain per (row group, k-chunk); drain-ADD into f32
// outbuf so GK may exceed the L1-resident A budget (KC chunking).
// HARD RULE (learned the hard way = the SMU wedge): per tile
//   pbuf(<=ABYTES_C) + g_outbuf(MM*GW*4) + g_d(GW*4) <= 32768 B.
// MM=8 fixed on device (M=16 doubles A -> overflow -> wedge).
// Stream: [0] meta {i32 rg0 | f32 s[MM] | f32 d[GW]};
//         per chunk c: CX objs A_c, then NWTc objs of W tiles (64 B each).
// A chunks: [2rb*KTC + kt] = ah, [(2rb+1)*KTC + kt] = al, KTC = KC/8.
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
#ifndef KC
#define KC GK
#endif

using MMUL = ::aie::mmul<8, 8, 8, int8_t, int8_t, acc32>;

namespace {
constexpr int KCH = GK / KC;                // k-chunks
constexpr int KTC = KC / 8;                 // k-tiles per chunk
constexpr int RG = GW / 8;
constexpr int M8 = MM / 8;
constexpr int TILES = RG * KTC;             // w-tiles per chunk
constexpr int TPO = (OB / 64) / 4 * 4;      // 52, block-aligned
constexpr int NWT = (TILES + TPO - 1) / TPO;
constexpr int ABYTES = 2 * M8 * KTC * 64;   // per chunk (hi+lo)
constexpr int CX = (ABYTES + OB - 1) / OB;
constexpr int NOD = (MM * GW * 4 + OB - 1) / OB;
constexpr int NOBJ = 1 + KCH * (CX + NWT);
}

static int g_rg0;
static float g_s[MM];
static float g_d[GW];
static int g_cur;
__attribute__((aligned(512))) static float g_outbuf[MM * GW];

extern "C" {

// fold-ADD row block rb, cols rl*8..+7:  out += (256*accH + accL)*(s*d)
static inline void fold_rb(float *out, ::aie::accum<acc32, 64> &acH,
                           ::aie::accum<acc32, 64> &acL, int rb, int rl) {
    const ::aie::vector<int32_t, 32> h0 = acH.extract<32>(0).to_vector<int32_t>();
    const ::aie::vector<int32_t, 32> h1 = acH.extract<32>(1).to_vector<int32_t>();
    const ::aie::vector<int32_t, 32> l0 = acL.extract<32>(0).to_vector<int32_t>();
    const ::aie::vector<int32_t, 32> l1 = acL.extract<32>(1).to_vector<int32_t>();
    const int32_t *a0 = (const int32_t *)&h0, *a1 = (const int32_t *)&h1;
    const int32_t *b0 = (const int32_t *)&l0, *b1 = (const int32_t *)&l1;
    for (int m = 0; m < 8; m++) {
        const float sm = g_s[rb * 8 + m];
        const int32_t *pa = m < 4 ? a0 + m * 8 : a1 + (m - 4) * 8;
        const int32_t *pb = m < 4 ? b0 + m * 8 : b1 + (m - 4) * 8;
        float *o = out + (rb * 8 + m) * GW + rl * 8;
        for (int n = 0; n < 8; n++)
            o[n] += ((float)pa[n] * 256.f + (float)pb[n]) * (sm * g_d[rl * 8 + n]);
    }
}

__attribute__((noinline)) void w8a16_reset() { g_cur = 0; }

__attribute__((noinline)) void w8a16_step(const uint8_t *__restrict obj,
                                        uint8_t *__restrict pbuf) {
    const int o = g_cur++;
    if (o == 0) {
        __builtin_memcpy(&g_rg0, obj, 4);
        __builtin_memcpy(g_s, obj + 4, MM * 4);
        __builtin_memcpy(g_d, obj + 4 + 4 * MM, GW * 4);
        for (int i = 0; i < MM * GW; i++) g_outbuf[i] = 0.f;
        return;
    }
    const int p = o - 1;                       // obj pos past meta
    const int pc = p % (CX + NWT);             // within-chunk position
    const int c = p / (CX + NWT);              // chunk index
    if (pc < CX) {                             // A chunk byte-copy
        const int off = pc * OB;
        const int n = ABYTES - off < OB ? ABYTES - off : OB;
        __builtin_memcpy(pbuf + off, obj, n);
        return;
    }
    const int8_t *A = reinterpret_cast<const int8_t *>(pbuf);
    const int t0 = (pc - CX) * TPO;
    const int nt = TILES - t0 < TPO ? TILES - t0 : TPO;
    static MMUL acH[M8], acL[M8];
    for (int t = t0; t < t0 + nt; t++) {
        const int rg = g_rg0 + t / KTC, kt = t % KTC;
        const auto wv = ::aie::load_v<64>(reinterpret_cast<const int8_t *>(obj) + (t - t0) * 64);
        for (int rb = 0; rb < M8; rb++) {
            acH[rb].zero = (kt == 0);
            acH[rb].mac(::aie::load_v<64>(A + (2 * rb * KTC + kt) * 64), wv);
            acL[rb].zero = (kt == 0);
            acL[rb].mac(::aie::load_v<64>(A + ((2 * rb + 1) * KTC + kt) * 64), wv);
        }
        if (kt == KTC - 1) {                   // chunk row group complete
            for (int rb = 0; rb < M8; rb++) {
                ::aie::accum<acc32, 64> h = acH[rb].to_accum();
                ::aie::accum<acc32, 64> l = acL[rb].to_accum();
                fold_rb(g_outbuf, h, l, rb, rg - g_rg0);
            }
        }
    }
    if (c == KCH - 1 && o == NOBJ - 1) return;  // last obj: flush will send
}

__attribute__((noinline)) void w8a16_flush(uint8_t *__restrict obj,
                                         const uint8_t *__restrict pbuf) {
    static int g_od = 0;
    const int total = MM * GW * 4;
    const int off = g_od * OB;
    const int n = total - off < OB ? total - off : OB;
    __builtin_memcpy(obj, (const uint8_t *)g_outbuf + off, n);
    if (++g_od >= NOD) g_od = 0;
}

#define A16_WRAP(i)                                                 \
    void w8a16_reset##i(void) { g_cur = 0; }                          \
    void w8a16_step##i(const uint8_t *o, uint8_t *p) { w8a16_step(o, p); } \
    void w8a16_flush##i(uint8_t *o, const uint8_t *p) { w8a16_flush(o, p); }
A16_WRAP(0) A16_WRAP(1) A16_WRAP(2) A16_WRAP(3)
A16_WRAP(4) A16_WRAP(5) A16_WRAP(6) A16_WRAP(7)

}
