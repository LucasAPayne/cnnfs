#pragma once

#include "util/types.h"

typedef struct vec_u8
{
    usize elements;
    u8* data;
} vec_u8;

vec_u8 vec_u8_init(usize elements, u8* data);
vec_u8 vec_u8_zeros(usize elements);

// Create vector of a given size filled with fill_value
vec_u8 vec_u8_full(usize elements, u8 fill_value);

u8 vec_u8_at(vec_u8 v, usize index);

// Set multiple vector values
void vec_u8_set_range(vec_u8* v, vec_u8 data, usize begin);

void vec_u8_print(vec_u8 v);

vec_u8 vec_u8_add(vec_u8 a, vec_u8 b);
vec_u8 vec_u8_scale(vec_u8 v, u8 a);

// Element-wise (Hadamard product)
// Vectors must be equal size
vec_u8 vec_u8_had(vec_u8 a, vec_u8 b);
