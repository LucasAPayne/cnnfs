#pragma once

#include "util/types.h"

typedef struct vec_f32
{
    usize elements;
    f32* data;
} vec_f32;

vec_f32 vec_f32_init(usize elements, f32* data);
vec_f32 vec_f32_zeros(usize elements);

// Create vector of a given size filled with fill_value
vec_f32 vec_f32_full(usize elements, f32 fill_value);

f32 vec_f32_at(vec_f32 v, usize index);

// Set multiple vector values
void vec_f32_set_range(vec_f32* v, vec_f32 data, usize begin);

void vec_f32_print(vec_f32 v);

vec_f32 vec_f32_add(vec_f32 a, vec_f32 b);
vec_f32 vec_f32_scale(vec_f32 v, f32 a);

// Element-wise (Hadamard product)
// Vectors must be equal size
vec_f32 vec_f32_had(vec_f32 a, vec_f32 b);
