#pragma once

#include "math/cnnfs_math.h"
#include "types.h"

void rand_seed(u64 seed);

// TODO(lucas): Support random integers
// TODO(lucas): Support 64-bit RNG

f32 rand_f32_uniform(f32 min, f32 max);
f32 rand_f32_gauss(f32 mean, f32 std_dev);
f32 rand_f32_gauss_standard(void);
