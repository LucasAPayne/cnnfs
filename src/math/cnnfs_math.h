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

inline vec_f32 sin_vec_f32(vec_f32 v)
{
    vec_f32 result = vec_f32_zeros(v.elements);
    for (usize i = 0; i < v.elements; ++i)
        result.data[i] = sin_f32(v.data[i]);
    
    return result;
}

inline vec_f32 cos_vec_f32(vec_f32 v)
{
    vec_f32 result = vec_f32_zeros(v.elements);
    for (usize i = 0; i < v.elements; ++i)
        result.data[i] = cos_f32(v.data[i]);
    
    return result;
}
