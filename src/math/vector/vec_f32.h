#pragma once

#include "util/types.h"

typedef struct vec_f32
{
    usize elements;
    f32* data;
} vec_f32;

vec_f32 vec_f32_init(usize elements, f32* data);
vec_f32 vec_f32_zeros(usize elements);

f32 vec_f32_at(vec_f32 v, usize index);

void vec_f32_print(vec_f32 m);

vec_f32 vec_f32_add(vec_f32 a, vec_f32 b);
vec_f32 vec_f32_scale(vec_f32 v, f32 a);
