#include "cnnfs_math.h"

#include <stdlib.h>

f32* linspace(f32 x1, f32 x2, usize n)
{
    f32* result = calloc(n, sizeof(f32));
    f32 dx = (x2 - x1) / (n - 1.0f);

    for (usize i = 0; i < n; ++i)
        result[i] = x1 + ((f32)i * dx);
    
    return result;
}
