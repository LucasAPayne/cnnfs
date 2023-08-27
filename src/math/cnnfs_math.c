#include "cnnfs_math.h"

#include <stdlib.h>

vec_f32 linspace(f32 x1, f32 x2, usize n)
{
    vec_f32 result = vec_f32_zeros(n);
    f32 dx = (x2 - x1) / (n - 1.0f);

    for (usize i = 0; i < n; ++i)
        result.data[i] = x1 + ((f32)i * dx);
    
    return result;
}
