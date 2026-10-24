/* SPU_FLOAT_RZ: spu_run_with_halt runs SPU code with FPCR at round toward zero with
 * flush-to-zero and puts the caller's FPCR back on the way out -- after a normal return, after a
 * halt (longjmp), and around a nested run. A run can happen synchronously on a PPU thread, whose
 * floating point must keep round to nearest with denormals. Without SPU_FLOAT_RZ=1 FPCR is never
 * touched. arm64 (FPCR), and x86 with AVX2 + FMA (MXCSR and g_spu_float_rz, below). */
#include "../spu_helpers.h"
#include "../spu_interp.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint8_t* vm_base;
uint32_t ppu_vm_size;
extern int spu_run_with_halt(void (*)(spu_context*), spu_context*);
extern void spu_halt(spu_context*);

#if defined(__aarch64__) && defined(__ARM_NEON) && !defined(SPU_SCALAR_HELPERS)
static uint64_t fpcr_get(void) { uint64_t v; __asm__ volatile("mrs %0, fpcr" : "=r"(v)); return v; }
static void fpcr_set(uint64_t v) { __asm__ volatile("msr fpcr, %0" :: "r"(v)); }

static spu_context ctx, inner_ctx;
static uint64_t seen_inner, seen_outer_after;
static int rz;

static void leaf_return(spu_context* c) { (void)c; seen_inner = fpcr_get(); }
static void leaf_halt(spu_context* c) { seen_inner = fpcr_get(); c->status = SPU_STATUS_STOPPED_BY_STOP; spu_halt(c); }
static void outer_nested(spu_context* c)
{
    (void)c;
    spu_context_init(&inner_ctx, 1);
    spu_run_with_halt(leaf_halt, &inner_ctx);
    seen_outer_after = fpcr_get();
}

static void run(void (*entry)(spu_context*), uint64_t caller, const char* what)
{
    fpcr_set(caller);
    seen_inner = seen_outer_after = ~0ull;
    spu_context_init(&ctx, 0);
    spu_run_with_halt(entry, &ctx);
    const uint64_t after = fpcr_get();
    fpcr_set(0);
    const uint64_t want_inner = rz ? SPU_FPCR_RZFZ : caller;
    if (seen_inner != want_inner || after != caller ||
        (entry == outer_nested && seen_outer_after != want_inner)) {
        fprintf(stderr, "FAIL %s (SPU_FLOAT_RZ=%d, caller FPCR %llX): inside %llX (want %llX), "
                "after %llX, outer after nested %llX\n", what, rz, (unsigned long long)caller,
                (unsigned long long)seen_inner, (unsigned long long)want_inner,
                (unsigned long long)after, (unsigned long long)seen_outer_after);
        exit(1);
    }
}

int main(void)
{
    const char* e = getenv("SPU_FLOAT_RZ");
    rz = e && !strcmp(e, "1");
    const uint64_t callers[] = { 0, 0x00400000ull /* round toward +inf */, SPU_FPCR_RZFZ };
    for (unsigned i = 0; i < sizeof callers / sizeof callers[0]; i++) {
        run(leaf_return, callers[i], "return");
        run(leaf_halt, callers[i], "halt");
        run(outer_nested, callers[i], "nested");
    }
    printf("PASS: SPU_FLOAT_RZ=%d FPCR inside runs and restored after return, halt and nesting\n", rz);
    return 0;
}
#elif defined(SPU_FLOAT_RZ_X86)
/* x86: MXCSR and g_spu_float_rz together. Inside a run with SPU_FLOAT_RZ=1, MXCSR is the
 * caller's with RC = toward zero, FTZ and DAZ (its exception masks kept) and the flag is 1; after
 * the run both are exactly the caller's, whatever MXCSR the caller had. A double op inside a run
 * puts the default MXCSR and flag 0 in for its own length and the run's mode back afterwards. */
static spu_context ctx, inner_ctx;
static unsigned seen_inner, seen_outer_after, seen_after_double;
static int flag_inner, flag_outer_after, flag_after_double, rz;

static void leaf_return(spu_context* c) { (void)c; seen_inner = _mm_getcsr(); flag_inner = g_spu_float_rz; }
static void leaf_halt(spu_context* c)
{
    seen_inner = _mm_getcsr(); flag_inner = g_spu_float_rz;
    c->status = SPU_STATUS_STOPPED_BY_STOP; spu_halt(c);
}
static void outer_nested(spu_context* c)
{
    (void)c;
    spu_context_init(&inner_ctx, 1);
    spu_run_with_halt(leaf_halt, &inner_ctx);
    seen_outer_after = _mm_getcsr(); flag_outer_after = g_spu_float_rz;
}
static void leaf_double(spu_context* c)
{
    (void)c;
    seen_inner = _mm_getcsr(); flag_inner = g_spu_float_rz;
    u128 a, b; memset(&a, 0, 16); memset(&b, 0, 16);
    a._u32[0] = 0x3FF00000u; b._u32[0] = 0x00080000u;   /* 1.0 + a denormal double */
    volatile u128 r = spu_dfa(a, b); (void)r;
    seen_after_double = _mm_getcsr(); flag_after_double = g_spu_float_rz;
}

static void run(void (*entry)(spu_context*), unsigned caller, const char* what)
{
    _mm_setcsr(caller);
    seen_inner = seen_outer_after = seen_after_double = ~0u;
    flag_inner = flag_outer_after = flag_after_double = -1;
    spu_context_init(&ctx, 0);
    spu_run_with_halt(entry, &ctx);
    const unsigned after = _mm_getcsr();
    const int flag_after = g_spu_float_rz;
    _mm_setcsr(0x1F80u);
    const unsigned want_inner = rz ? (caller & ~0xE040u) | SPU_MXCSR_RZFZ_BITS : caller;
    const int bad = seen_inner != want_inner || flag_inner != rz || after != caller || flag_after != 0 ||
        (entry == outer_nested && (seen_outer_after != want_inner || flag_outer_after != rz)) ||
        (entry == leaf_double && (seen_after_double != want_inner || flag_after_double != rz));
    if (bad) {
        fprintf(stderr, "FAIL %s (SPU_FLOAT_RZ=%d, caller MXCSR %04X): inside %04X flag %d (want %04X flag %d), "
                "after %04X flag %d, outer after nested %04X flag %d, after a double op %04X flag %d\n",
                what, rz, caller, seen_inner, flag_inner, want_inner, rz, after, flag_after,
                seen_outer_after, flag_outer_after, seen_after_double, flag_after_double);
        exit(1);
    }
}

int main(void)
{
    const char* e = getenv("SPU_FLOAT_RZ");
    rz = e && !strcmp(e, "1");
    const unsigned callers[] = {
        0x1F80u,                          /* default: round to nearest, everything masked */
        0x5F80u,                          /* round toward +inf */
        0x9F80u,                          /* FTZ only */
        0x1D80u,                          /* divide-by-zero unmasked: the masks are the caller's */
        0x1F80u | SPU_MXCSR_RZFZ_BITS,    /* already round toward zero, FTZ, DAZ */
    };
    for (unsigned i = 0; i < sizeof callers / sizeof callers[0]; i++) {
        run(leaf_return, callers[i], "return");
        run(leaf_halt, callers[i], "halt");
        run(outer_nested, callers[i], "nested");
        run(leaf_double, callers[i], "double op");
    }
    printf("PASS: SPU_FLOAT_RZ=%d MXCSR and flag inside runs and restored after return, halt, "
           "nesting and double ops (x86)\n", rz);
    return 0;
}
#else
int main(void) { puts("SKIP: SPU_FLOAT_RZ needs arm64, or x86 with AVX2 + FMA"); return 0; }
#endif

/* Link-only boundaries outside this CPU-only fixture. Fail if any is reached. */
void spu_thread_publish_ctx(uint32_t tid, void* c) { (void)tid; (void)c; abort(); }
void ppu_resv_break_store(uint64_t ea) { (void)ea; abort(); }
void ps3_ww_report_inline(uint32_t a, uint64_t v, int w) { (void)a; (void)v; (void)w; abort(); }
unsigned long long ps3_ms_now(void) { abort(); }
uint32_t g_ww_lo = 0xFFFFFFFFu, g_ww_hi;
int g_resv_store_active;
uint32_t g_barrier_sync_watch;
uint32_t g_ydkj_real_spurs_ea, g_ydkj_real_taskset_ea, g_ydkj_real_taskid;
void spurs_ef_set_from_spu(uint32_t ea, uint16_t bits) { (void)ea; (void)bits; abort(); }
void ydkj_wake_all_event_flags(void) { abort(); }
void spurs_pm_build_context(spu_context* c, uint32_t a, uint32_t b, uint32_t d) {
    (void)c; (void)a; (void)b; (void)d; abort();
}
uint32_t g_spu_image_src_ea, g_spu_image_ls_start, g_spu_image_span;
