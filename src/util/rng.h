#pragma once

#include "util/types.h"

void rand_seed(u64 seed);
f32 rand_f32_uniform();
f32 rand_f32_gauss(f32 mean, f32 std_dev);
