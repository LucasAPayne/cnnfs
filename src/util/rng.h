#pragma once

#include "math/cnnfs_math.h"
#include "util/types.h"

void rand_seed(u64 seed);

// Generate single random numbers
f32 rand_f32_uniform(f32 min, f32 max);
f32 rand_f32_gauss(f32 mean, f32 std_dev);
f32 rand_f32_gauss_standard(void);

// Generate sequences of n random numbers
vec_f32 rand_vec_f32_uniform(f32 min, f32 max, usize n);
vec_f32 rand_vec_f32_gauss(f32 mean, f32 std_dev, usize n);
vec_f32 rand_vec_f32_gauss_standard(usize n);

mat_f32 rand_mat_f32_uniform(f32 min, f32 max, usize rows, usize cols);
mat_f32 rand_mat_f32_gauss(f32 mean, f32 std_dev, usize rows, usize cols);
mat_f32 rand_mat_f32_gauss_standard(usize rows, usize cols);
