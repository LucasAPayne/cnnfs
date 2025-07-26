#include "profile.h"
#include "types.h"

#include <stdio.h>

#if PROFILER
ProfileAnchor global_profiler_anchors[NUM_PROFILER_ANCHORS];
u32 global_profiler_parent;
#endif

Profiler global_profiler;

#if _WIN32
    #include <intrin.h>
    #include <windows.h>

    internal u64 get_os_timer_freq()
    {
        LARGE_INTEGER freq = {0};
        QueryPerformanceFrequency(&freq);
        return freq.QuadPart;
    }

    internal u64 read_os_timer()
    {
        LARGE_INTEGER value = {0};
        QueryPerformanceCounter(&value);
        return value.QuadPart;
    }
#endif

#if PROFILER
internal void print_anchor_stats(u64 total_elapsed_tsc, u64 cpu_freq)
{
    for (u32 i = 0; i < countof(global_profiler_anchors); ++i)
    {
        ProfileAnchor* anchor = global_profiler_anchors + i;
        if (anchor->label)
        {
            f64 percent = 100.0 * (f64)anchor->tsc_elapsed_exclusive / (f64)total_elapsed_tsc;
            f64 ms_elapsed = get_ms_elapsed(anchor->tsc_elapsed_exclusive, cpu_freq);
            printf("  %s[%llu]: %.4fms (%.2f%%", anchor->label, anchor->hit_count, ms_elapsed, percent);
            if (anchor->tsc_elapsed_exclusive != anchor->tsc_elapsed_inclusive)
            {
                f64 percent_with_children = 100.0 * (f64)anchor->tsc_elapsed_inclusive / (f64)total_elapsed_tsc;
                printf(", %.2f%% w/ children", percent_with_children);
            }
            printf(")\n");
        }
    }
}
#endif

u64 estimate_cpu_freq()
{
    u64 ms_to_wait = 100;
    u64 os_freq = get_os_timer_freq();
    u64 os_wait_time = os_freq * ms_to_wait / 1000;

    u64 cpu_start = read_cpu_timer();
    u64 os_start = read_os_timer();
    u64 os_elapsed = 0;
    u64 os_end = 0;
    while (os_elapsed < os_wait_time)
    {
        os_end = read_os_timer();
        os_elapsed = os_end - os_start;
    }
    u64 cpu_end = read_cpu_timer();
    u64 cpu_elapsed = cpu_end - cpu_start;

    u64 cpu_freq = 0;
    if (os_elapsed)
        cpu_freq = os_freq * cpu_elapsed / os_elapsed;

    return cpu_freq;
}

u64 read_cpu_timer()
{
    return __rdtsc();
}

f64 get_seconds_elapsed(u64 tsc_elapsed, u64 freq)
{
    f64 result = (f64)tsc_elapsed / (f64)freq;
    return result;
}

f64 get_ms_elapsed(u64 tsc_elapsed, u64 freq)
{
    f64 result = get_seconds_elapsed(tsc_elapsed, freq) * 1000.0;
    return result;
}

void profiler_begin()
{
    global_profiler.anchor_index = 1;
    global_profiler.tsc_start = read_cpu_timer();
}

void profiler_end()
{
    global_profiler.tsc_end = read_cpu_timer();
}

void profiler_print()
{
    u64 cpu_freq = estimate_cpu_freq();
    u64 total_elapsed_tsc = global_profiler.tsc_end - global_profiler.tsc_start;
    f64 ms_elapsed = get_ms_elapsed(total_elapsed_tsc, cpu_freq);

    if (cpu_freq)
        printf("Total time: %.4fms (%.4fs)\n", ms_elapsed, ms_elapsed / 1000.0);

    print_anchor_stats(total_elapsed_tsc, cpu_freq);
}
