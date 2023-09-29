#include "vec_u8.h"

#include <stdio.h>
#include <stdlib.h>

vec_u8 vec_u8_init(usize elements, u8* data)
{
    vec_u8 result = {0};
    result.elements = elements;
    result.data = data;

    return result;
}

vec_u8 vec_u8_zeros(usize elements)
{
    vec_u8 result = {0};
    result.elements = elements;

    result.data = calloc(elements, sizeof(u8));
    ASSERT(result.data);

    return result;
}

vec_u8 vec_u8_full(usize elements, u8 fill_value)
{
    vec_u8 v = vec_u8_zeros(elements);

    for (usize i = 0; i < v.elements; ++i)
        v.data[i] = fill_value;

    return v;
}

u8 vec_u8_at(vec_u8 v, usize index)
{
    u8 result = 0;
    result = v.data[index];
    return result;
}

void vec_u8_set_range(vec_u8* v, vec_u8 data, usize begin)
{
    // Ensure there is enough room in the vector
    ASSERT(v->elements >= data.elements + begin);

    for (usize i = 0; i < data.elements; ++i)
        v->data[begin+i] = data.data[i];
}

void vec_u8_print(vec_u8 v)
{
    printf("[");
    for (usize i = 0; i < v.elements; ++i)
    {
        if (i != 0) printf(", ");
        printf("%hhu", v.data[i]);
    }
    printf("]\n");
}

vec_u8 vec_u8_add(vec_u8 a, vec_u8 b)
{
    ASSERT(a.elements == b.elements);
    vec_u8 result = vec_u8_zeros(a.elements);

    for (usize i = 0; i < result.elements; ++i)
        result.data[i] = a.data[i] + b.data[i];

    return result;
}

vec_u8 vec_u8_scale(vec_u8 v, u8 a)
{
    vec_u8 result = vec_u8_zeros(v.elements);

    for (usize i = 0; i < result.elements; ++i)
        result.data[i] = a*v.data[i];

    return result;
}

vec_u8 vec_u8_had(vec_u8 a, vec_u8 b)
{
    ASSERT(a.elements == b.elements);
    vec_u8 result = vec_u8_zeros(a.elements);

    for (usize i = 0; i < result.elements; ++i)
        result.data[i] = a.data[i]*b.data[i];

    return result;
}
