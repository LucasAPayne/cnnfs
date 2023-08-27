#pragma once

#include "math/cnnfs_math.h"
#include "util/types.h"

void rand_seed(u64 seed);

// Generate single random numbers
f32 rand_f32_uniform(f32 min, f32 max);
f32 rand_f32_gauss(f32 mean, f32 std_dev);

// Generate sequences of n random numbers
vec_f32 randn_f32_uniform(f32 min, f32 max, usize n);
vec_f32 randn_f32_gauss(f32 mean, f32 std_dev, usize n);
