/* IMPORTANT(lucas): This profiler will only work for unity (single translation unit) builds,
 * due to the use of __COUNTER__. It used to assert at compile time that the number of points
 * does not exceed the size of the Profiler::anchors array.
 */

/* PLAN(lucas):
 * Check that a time_*_begin() always has a matching time_*_end() and vice versa
 * Support multithreading
 * More tabular print output
 * Track more information (inclusive/exclusive time, time per call)
 * Sort by metric
 * Output to file option (txt or csv format)
 */

#pragma once

#include "types.h"

struct Profiler
{
    u64 tsc_start;
    u64 tsc_end;
    u32 anchor_index;
};

extern Profiler global_profiler;

u64 estimate_cpu_freq();
u64 read_cpu_timer();

f64 get_seconds_elapsed(u64 tsc_elapsed, u64 freq);
f64 get_ms_elapsed(u64 tsc_elapsed, u64 freq);

void profiler_begin();
void profiler_end();
void profiler_print();

#ifndef PROFILER
#define PROFILER 0
#endif

#if PROFILER

struct ProfileAnchor
{
    u64 tsc_start;
    u64 old_tsc_elapsed_inclusive;
    u32 parent_index;

    u64 tsc_elapsed_exclusive; // NOTE(lucas): Does NOT include children
    u64 tsc_elapsed_inclusive; // NOTE(lucas): DOES include children
    u64 hit_count;
    const char* label;
};

#define NUM_PROFILER_ANCHORS 4096
extern ProfileAnchor global_profiler_anchors[NUM_PROFILER_ANCHORS];
extern u32 global_profiler_parent;

inline void profile_block_begin(const char* name, u32 anchor_index)
{
    global_profiler.anchor_index = anchor_index;

    ProfileAnchor* anchor = global_profiler_anchors + global_profiler.anchor_index;
    anchor->parent_index = global_profiler_parent;
    anchor->label = name;
    anchor->old_tsc_elapsed_inclusive = anchor->tsc_elapsed_inclusive;

    global_profiler_parent = global_profiler.anchor_index;
    anchor->tsc_start = read_cpu_timer();
}

inline void profile_block_end()
{
    ProfileAnchor* anchor = global_profiler_anchors + global_profiler.anchor_index;
    u64 elapsed = read_cpu_timer() - anchor->tsc_start;
    global_profiler_parent = anchor->parent_index;

    ProfileAnchor* parent = global_profiler_anchors + global_profiler_parent;

    parent->tsc_elapsed_exclusive -= elapsed;
    anchor->tsc_elapsed_exclusive += elapsed;
    anchor->tsc_elapsed_inclusive = anchor->old_tsc_elapsed_inclusive + elapsed;
    ++anchor->hit_count;

    global_profiler.anchor_index = global_profiler_parent;
}

#define time_block_begin(name) profile_block_begin((name), __COUNTER__+1)
#define time_block_end() profile_block_end()

#define time_function_begin() time_block_begin(__func__)
#define time_function_end() time_block_end()

#define profiler_static_assert() static_assert(__COUNTER__ < countof(global_profiler_anchors), \
              "Number of profile points exceeds size of Profiler::Anchors array")

#else

#define time_block_begin(...)
#define time_block_end()

#define time_function_begin()
#define time_function_end()

#define print_anchor_stats(...)
#define profiler_static_assert()

#endif
