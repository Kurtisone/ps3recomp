/* SPU_FLOAT_RZ on x86: with MXCSR at RZ+FTZ+DAZ and g_spu_float_rz set, the native
 * spu_fa/fs/fm/fma/fms/fnms must be bit-identical to the exact integer references (spu_*_ref)
 * for every input: random words, ordinary floats, the normal/extended-range boundaries, zeros
 * and denormals, mixed per lane. Double precision, frds and the double compares must give the
 * same answer as with the default MXCSR, and leave MXCSR as they found it.
 * Build on x86 with AVX2:
 *   cc -O2 -mavx2 -mfma -I../../../include -I.. test_spu_float_rz.c -o t -lm && ./t */
#include "../spu_helpers.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(SPU_FLOAT_RZ_X86)
static uint64_t s_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd32(void)
{
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return (uint32_t)(s_rng >> 16);
}

static uint32_t rnd_float(void)
{
    uint32_t sign = (rnd32() & 1u) << 31, man = rnd32() & 0x7FFFFFu, e;
    switch (rnd32() % 10) {
    case 0: e = 0; break;                              /* zero / denormal */
    case 1: e = 1 + rnd32() % 4; break;                /* bottom of normal range */
    case 2: e = 248 + rnd32() % 8; break;              /* top, into extended range */
    case 3: return rnd32();                            /* anything */
    case 4: return sign;                               /* +-0 */
    case 5: e = 60 + rnd32() % 8; break;               /* products near the bottom */
    default: e = 110 + rnd32() % 36; break;            /* ordinary */
    }
    return sign | (e << 23) | man;
}

static uint64_t rnd_double(void)
{
    const uint64_t sign = (uint64_t)(rnd32() & 1u) << 63, man = ((uint64_t)rnd32() << 32 | rnd32()) & 0xFFFFFFFFFFFFFull;
    uint64_t e;
    switch (rnd32() % 6) {
    case 0: e = 0; break;                              /* zero / denormal */
    case 1: e = 1 + rnd32() % 4; break;
    case 2: e = 2040 + rnd32() % 8; break;             /* near overflow, Inf/NaN */
    case 3: return (uint64_t)rnd32() << 32 | rnd32();
    default: e = 1000 + rnd32() % 48; break;
    }
    return sign | e << 52 | man;
}

static long s_bad;
static void report(const char* op, u128 got, u128 want)
{
    if (s_bad++ < 20)
        printf("MISMATCH %s got %08X %08X %08X %08X want %08X %08X %08X %08X\n", op,
               got._u32[0], got._u32[1], got._u32[2], got._u32[3],
               want._u32[0], want._u32[1], want._u32[2], want._u32[3]);
}
static void cmp(const char* op, u128 got, u128 want) { if (memcmp(got._u32, want._u32, 16)) report(op, got, want); }

static void rz_on(void)  { _mm_setcsr((_mm_getcsr() & ~SPU_MXCSR_RZ_FTZ_DAZ) | SPU_MXCSR_RZ_FTZ_DAZ); g_spu_float_rz = 1; }
static void rz_off(unsigned m) { g_spu_float_rz = 0; _mm_setcsr(m); }
#endif

int main(void)
{
#if !defined(SPU_FLOAT_RZ_X86)
    puts("test_spu_float_rz: no x86 AVX2 RZ mode in this build, nothing to compare");
    return 0;
#else
    const unsigned def = _mm_getcsr();
    const long N = 4000000;

    /* single precision: native forms under RZ+FTZ+DAZ against the integer references */
    rz_on();
    for (long it = 0; it < N; it++) {
        u128 a, b, c;
        for (int k = 0; k < 4; k++) { a._u32[k] = rnd_float(); b._u32[k] = rnd_float(); c._u32[k] = rnd_float(); }
        cmp("fa", spu_fa(a, b), spu_fa_ref(a, b));
        cmp("fs", spu_fs(a, b), spu_fs_ref(a, b));
        cmp("fm", spu_fm(a, b), spu_fm_ref(a, b));
        cmp("fma", spu_fma(a, b, c), spu_fma_ref(a, b, c));
        cmp("fms", spu_fms(a, b, c), spu_fms_ref(a, b, c));
        cmp("fnms", spu_fnms(a, b, c), spu_fnms_ref(a, b, c));
    }
    rz_off(def);
    printf("single precision under RZ: %ld mismatches\n", s_bad);

    /* double precision: under the mode, the same answer as with the default MXCSR */
    const long bad_single = s_bad;
    for (long it = 0; it < N / 4; it++) {
        u128 a, b, c;
        for (int k = 0; k < 2; k++) {
            const uint64_t x = rnd_double(), y = rnd_double(), z = rnd_double();
            a._u32[2*k] = (uint32_t)(x >> 32); a._u32[2*k+1] = (uint32_t)x;
            b._u32[2*k] = (uint32_t)(y >> 32); b._u32[2*k+1] = (uint32_t)y;
            c._u32[2*k] = (uint32_t)(z >> 32); c._u32[2*k+1] = (uint32_t)z;
        }
        const u128 w[12] = { spu_dfa(a, b), spu_dfs(a, b), spu_dfm(a, b), spu_dfma(a, b, c), spu_dfms(a, b, c),
                             spu_dfnms(a, b, c), spu_dfnma(a, b, c), spu_frds(a), spu_dfceq(a, b), spu_dfcmeq(a, b),
                             spu_dfcgt(a, b), spu_dfcmgt(a, b) };
        rz_on();
        const unsigned in_mode = _mm_getcsr();
        const u128 g[12] = { spu_dfa(a, b), spu_dfs(a, b), spu_dfm(a, b), spu_dfma(a, b, c), spu_dfms(a, b, c),
                             spu_dfnms(a, b, c), spu_dfnma(a, b, c), spu_frds(a), spu_dfceq(a, b), spu_dfcmeq(a, b),
                             spu_dfcgt(a, b), spu_dfcmgt(a, b) };
        if ((_mm_getcsr() & ~0x3Fu) != (in_mode & ~0x3Fu)) { if (s_bad++ < 20) puts("MXCSR changed by a double op"); }
        rz_off(def);
        static const char* names[12] = { "dfa", "dfs", "dfm", "dfma", "dfms", "dfnms", "dfnma", "frds",
                                         "dfceq", "dfcmeq", "dfcgt", "dfcmgt" };
        for (int k = 0; k < 12; k++) cmp(names[k], g[k], w[k]);
    }
    printf("double precision under RZ: %ld mismatches\n", s_bad - bad_single);

    /* speed: fma + fm, exact path vs native */
    u128 in[64];
    for (int i = 0; i < 64; i++)
        for (int k = 0; k < 4; k++) in[i]._u32[k] = (rnd32() & 0x80000000u) | ((120 + rnd32() % 16) << 23) | (rnd32() & 0x7FFFFFu);
    for (int mode = 0; mode < 2; mode++) {
        if (mode) rz_on();
        u128 acc = spu_zero();
        const clock_t t0 = clock();
        for (long i = 0; i < 20000000; i++) {
            const u128 x = spu_fma(in[i & 63], in[(i + 5) & 63], in[(i + 9) & 63]), y = spu_fm(in[(i + 3) & 63], in[(i + 11) & 63]);
            acc._u64[0] ^= x._u64[0] ^ y._u64[1]; acc._u64[1] += x._u64[1];
        }
        const clock_t t1 = clock();
        if (mode) rz_off(def);
        printf("fma+fm %s: %.2f ns (%u)\n", mode ? "native RZ" : "exact    ",
               1e9 * (double)(t1 - t0) / CLOCKS_PER_SEC / 20000000, (unsigned)(acc._u32[0] & 1));
    }
    return s_bad ? 1 : 0;
#endif
}
