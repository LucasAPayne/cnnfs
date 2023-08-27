#pragma once

#include "util/types.h"
#include "matrix/mat_f32.h"
#include "matrix/mat_u8.h"
#include "vector/vec_f32.h"

#include <math.h>

// Generate a sequence of n evenly-spaced numbers in the range [x1, x2]
vec_f32 linspace(f32 x1, f32 x2, usize n);

inline f32 sin_f32(f32 x)
{
    f32 result = sinf(x);
    return result;
}

inline f32 cos_f32(f32 x)
{
    f32 result = cosf(x);
    return result;
}
