/* IMPORTANT: Randomly generated sequences may not be reproducible across CPU and GPU implementations,
 * or across different versions or commits of this library.
*/

#pragma once

#include "math/cnnfs_math.h"
#include "types.h"

internal void rand_seed(u64 seed);

// TODO(lucas): Support random integers
// TODO(lucas): Support 64-bit RNG

internal f32 rand_f32_uniform(f32 min, f32 max);
internal f32 rand_f32_gauss(f32 mean, f32 std_dev);
internal f32 rand_f32_gauss_standard();
