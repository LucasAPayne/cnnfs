#include "vec_f32.h"

#include <stdio.h>
#include <stdlib.h>

vec_f32 vec_f32_init(usize elements, f32* data)
{
    vec_f32 result = {0};
    result.elements = elements;
    result.data = data;

    return result;
}

vec_f32 vec_f32_zeros(usize elements)
{
    vec_f32 result = {0};
    result.elements = elements;

    result.data = calloc(elements, sizeof(f32));
    ASSERT(result.data);

    return result;
}

f32 vec_f32_at(vec_f32 v, usize index)
{
    f32 result = 0.0f;
    result = v.data[index];
    return result;
}

void vec_f32_print(vec_f32 v)
{
    printf("[");
    for (usize i = 0; i < v.elements; ++i)
    {
        if (i != 0) printf(", ");
        printf("%f", v.data[i]);
    }
    printf("]\n");
}

vec_f32 vec_f32_add(vec_f32 a, vec_f32 b)
{
    ASSERT(a.elements == b.elements);
    vec_f32 result = vec_f32_zeros(a.elements);

    for (usize i = 0; i < result.elements; ++i)
        result.data[i] = a.data[i] + b.data[i];

    return result;
}

vec_f32 vec_f32_scale(vec_f32 v, f32 a)
{
    vec_f32 result = vec_f32_zeros(v.elements);

    for (usize i = 0; i < result.elements; ++i)
        result.data[i] = a*v.data[i];

    return result;
}
