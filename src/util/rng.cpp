#pragma once

/*
 * Random number generation functions for uniform and normal distributions
 * Normal (Gaussian) distribution numbers are generated using the Box-Muller method
 */

#include "rng.h"
#include <math.h> // sqrtf(), logf()

/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *       http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

/* NOTE(lucas): This is a trimmed version of the original.
 * Only the global RNG is used since it is treated as internal state,
 * and types and functions are renamed to match this project's style.
 */
struct PCG32RNG
{              // Internals are *Private*.
    u64 state; // RNG state.  All values are possible.
    u64 inc;   // Controls which RNG sequence (stream) is
               // selected. Must *always* be odd.
};

struct GPURNG
{
    u64 seed;
};

// Make RNG global here so RNG type can be opaque to user
#define PCG32_INITIALIZER {0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL}
global PCG32RNG pcg32_global = PCG32_INITIALIZER;
global GPURNG gpu_rng_global = {};

// TODO(lucas): In the last part of the computation, rot is a u32, so taking the negative does
// not make sense mathematically.
// Is there an equivalent operation that can replace this so that I can get rid of disabling this warning?
#pragma warning(push)
#pragma warning(disable: 4146)
/**
 * Generate a uniformly distributed 32-bit random number
 */
internal u32 pcg32_rand()
{
    u64 oldstate = pcg32_global.state;
    pcg32_global.state = oldstate * 6364136223846793005ULL + pcg32_global.inc;
    u32 xorshifted = (u32)(((oldstate >> 18u) ^ oldstate) >> 27u);
    u32 rot = oldstate >> 59u;
    u32 result = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return result;
}
#pragma warning(pop)

/**
 * Seed the rng.  Specified in two parts, state initializer and a
 * sequence selection constant (a.k.a. stream id)
 */
internal void pcg32_set_seed(u64 init_state, u64 init_seq)
{
    pcg32_global.state = 0U;
    pcg32_global.inc = (init_seq << 1u) | 1u;
    pcg32_rand();
    pcg32_global.state += init_state;
    pcg32_rand();
}

/*
 * NOTE(lucas): General RNG code starts here
*/

/**
 * Set seed of RNG
 */
internal void rand_seed(u64 seed)
{
    // NOTE(lucas): For now, only the global PCG32 RNG is being used.
    // Streams are not currently being used, so set to 0 for now.
    pcg32_set_seed(seed, 0);
    gpu_rng_global.seed = seed;
}

/**
 * Generate a random f32 in the range [min, max] from the uniform distribution
 */
internal f32 rand_f32_uniform(f32 min, f32 max)
{
    ASSERT(max > min, "The maximum value must be greater than the minimum value.\n");
    f32 result = (f32)pcg32_rand() / (f32)UINT32_MAX;
    result = min + result * (max - min);
    return result;
}

/**
 * Generate a random f32 from the Guassian (normal) distribution with a given mean and standard deviation.
 */
internal f32 rand_f32_gauss(f32 mean, f32 std_dev)
{
    /* NOTE(lucas): Currently, the Marsaglia polar method variation of the Box-Muller method is being used.
     * This algorithm works by randomly picking points (x,y) in the square -1 < x < 1, -1 < y < 1
     * until 0 < s = x^2 + y^2 < 1. To put the points into the normal distribution,
     * they are multiplied by sqrt(-2*ln(s)/s).
     * 
     * Ziggurat is probably a better algorithm to use here, but it is more complex. May be revisited later.
     * This algorithm generates random Gaussian numbers in pairs, so it keeps track of whether it has
     * already generated the next one and what the value of it is.
     */
    persist b32 has_gauss = false;
    persist f32 next_gauss = 0.0f;
    f32 result = next_gauss;

    // Only perform algorithm if there is no spare result
    if (has_gauss)
        has_gauss = false;
    else
    {
        f32 x, y; // Random point in square
        f32 s; // s = x^2 + y^2
        do
        {
            // Pick uniformly random points and move into the range [-1.0, 1.0]
            x = 2.0f*rand_f32_uniform(0.0f, 1.0f) - 1.0f;
            y = 2.0f*rand_f32_uniform(0.0f, 1.0f) - 1.0f;
            s = x*x + y*y;
        } while(s >= 1 || s == 0);

        // Define multiplier and generate Gauss pair
        f32 multiplier = sqrtf(-2*logf(s)/s);
        result = x*multiplier;
        next_gauss = y*multiplier;

        // Next function call will just return next_gauss
        has_gauss = true;
    }

    // Transform result based on desired mean and standard deviation
    result = result*std_dev + mean;
    return result;
}

internal f32 rand_f32_gauss_standard()
{
    f32 result = rand_f32_gauss(0.0f, 1.0f);
    return result;
}
