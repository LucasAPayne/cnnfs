#pragma once

#include "../util/types.h"

#include <stdio.h>

#include <stdlib.h>

typedef struct vec_u8
{
	usize elements;
	u8* data;
} vec_u8;

typedef struct vec_u16
{
	usize elements;
	u16* data;
} vec_u16;

typedef struct vec_u32
{
	usize elements;
	u32* data;
} vec_u32;

typedef struct vec_u64
{
	usize elements;
	u64* data;
} vec_u64;

typedef struct vec_i8
{
	usize elements;
	i8* data;
} vec_i8;

typedef struct vec_i16
{
	usize elements;
	i16* data;
} vec_i16;

typedef struct vec_i32
{
	usize elements;
	i32* data;
} vec_i32;

typedef struct vec_i64
{
	usize elements;
	i64* data;
} vec_i64;

typedef struct vec_f32
{
	usize elements;
	f32* data;
} vec_f32;

typedef struct vec_f64
{
	usize elements;
	f64* data;
} vec_f64;

vec_u8 vec_u8_init(usize elements, u8* data);
vec_u16 vec_u16_init(usize elements, u16* data);
vec_u32 vec_u32_init(usize elements, u32* data);
vec_u64 vec_u64_init(usize elements, u64* data);
vec_i8 vec_i8_init(usize elements, i8* data);
vec_i16 vec_i16_init(usize elements, i16* data);
vec_i32 vec_i32_init(usize elements, i32* data);
vec_i64 vec_i64_init(usize elements, i64* data);
vec_f32 vec_f32_init(usize elements, f32* data);
vec_f64 vec_f64_init(usize elements, f64* data);

vec_u8 vec_u8_zeros(usize elements);
vec_u16 vec_u16_zeros(usize elements);
vec_u32 vec_u32_zeros(usize elements);
vec_u64 vec_u64_zeros(usize elements);
vec_i8 vec_i8_zeros(usize elements);
vec_i16 vec_i16_zeros(usize elements);
vec_i32 vec_i32_zeros(usize elements);
vec_i64 vec_i64_zeros(usize elements);
vec_f32 vec_f32_zeros(usize elements);
vec_f64 vec_f64_zeros(usize elements);

vec_u8 vec_u8_full(usize elements, u8 fill_value);
vec_u16 vec_u16_full(usize elements, u16 fill_value);
vec_u32 vec_u32_full(usize elements, u32 fill_value);
vec_u64 vec_u64_full(usize elements, u64 fill_value);
vec_i8 vec_i8_full(usize elements, i8 fill_value);
vec_i16 vec_i16_full(usize elements, i16 fill_value);
vec_i32 vec_i32_full(usize elements, i32 fill_value);
vec_i64 vec_i64_full(usize elements, i64 fill_value);
vec_f32 vec_f32_full(usize elements, f32 fill_value);
vec_f64 vec_f64_full(usize elements, f64 fill_value);

void vec_u8_set_range(vec_u8* v, vec_u8 data, usize begin);
void vec_u16_set_range(vec_u16* v, vec_u16 data, usize begin);
void vec_u32_set_range(vec_u32* v, vec_u32 data, usize begin);
void vec_u64_set_range(vec_u64* v, vec_u64 data, usize begin);
void vec_i8_set_range(vec_i8* v, vec_i8 data, usize begin);
void vec_i16_set_range(vec_i16* v, vec_i16 data, usize begin);
void vec_i32_set_range(vec_i32* v, vec_i32 data, usize begin);
void vec_i64_set_range(vec_i64* v, vec_i64 data, usize begin);
void vec_f32_set_range(vec_f32* v, vec_f32 data, usize begin);
void vec_f64_set_range(vec_f64* v, vec_f64 data, usize begin);

void vec_u8_print(vec_u8 v);
void vec_u16_print(vec_u16 v);
void vec_u32_print(vec_u32 v);
void vec_u64_print(vec_u64 v);
void vec_i8_print(vec_i8 v);
void vec_i16_print(vec_i16 v);
void vec_i32_print(vec_i32 v);
void vec_i64_print(vec_i64 v);
void vec_f32_print(vec_f32 v);
void vec_f64_print(vec_f64 v);

vec_u8 vec_u8_add(vec_u8 a, vec_u8 b);
vec_u16 vec_u16_add(vec_u16 a, vec_u16 b);
vec_u32 vec_u32_add(vec_u32 a, vec_u32 b);
vec_u64 vec_u64_add(vec_u64 a, vec_u64 b);
vec_i8 vec_i8_add(vec_i8 a, vec_i8 b);
vec_i16 vec_i16_add(vec_i16 a, vec_i16 b);
vec_i32 vec_i32_add(vec_i32 a, vec_i32 b);
vec_i64 vec_i64_add(vec_i64 a, vec_i64 b);
vec_f32 vec_f32_add(vec_f32 a, vec_f32 b);
vec_f64 vec_f64_add(vec_f64 a, vec_f64 b);

vec_u8 vec_u8_scale(vec_u8 v, u8 a);
vec_u16 vec_u16_scale(vec_u16 v, u16 a);
vec_u32 vec_u32_scale(vec_u32 v, u32 a);
vec_u64 vec_u64_scale(vec_u64 v, u64 a);
vec_i8 vec_i8_scale(vec_i8 v, i8 a);
vec_i16 vec_i16_scale(vec_i16 v, i16 a);
vec_i32 vec_i32_scale(vec_i32 v, i32 a);
vec_i64 vec_i64_scale(vec_i64 v, i64 a);
vec_f32 vec_f32_scale(vec_f32 v, f32 a);
vec_f64 vec_f64_scale(vec_f64 v, f64 a);

vec_u8 vec_u8_had(vec_u8 a, vec_u8 b);
vec_u16 vec_u16_had(vec_u16 a, vec_u16 b);
vec_u32 vec_u32_had(vec_u32 a, vec_u32 b);
vec_u64 vec_u64_had(vec_u64 a, vec_u64 b);
vec_i8 vec_i8_had(vec_i8 a, vec_i8 b);
vec_i16 vec_i16_had(vec_i16 a, vec_i16 b);
vec_i32 vec_i32_had(vec_i32 a, vec_i32 b);
vec_i64 vec_i64_had(vec_i64 a, vec_i64 b);
vec_f32 vec_f32_had(vec_f32 a, vec_f32 b);
vec_f64 vec_f64_had(vec_f64 a, vec_f64 b);


inline u8 vec_u8_at(vec_u8 v, usize index)
{
	u8 result = 0;
	result = v.data[index];
	return result;
}

inline u16 vec_u16_at(vec_u16 v, usize index)
{
	u16 result = 0;
	result = v.data[index];
	return result;
}

inline u32 vec_u32_at(vec_u32 v, usize index)
{
	u32 result = 0;
	result = v.data[index];
	return result;
}

inline u64 vec_u64_at(vec_u64 v, usize index)
{
	u64 result = 0;
	result = v.data[index];
	return result;
}

inline i8 vec_i8_at(vec_i8 v, usize index)
{
	i8 result = 0;
	result = v.data[index];
	return result;
}

inline i16 vec_i16_at(vec_i16 v, usize index)
{
	i16 result = 0;
	result = v.data[index];
	return result;
}

inline i32 vec_i32_at(vec_i32 v, usize index)
{
	i32 result = 0;
	result = v.data[index];
	return result;
}

inline i64 vec_i64_at(vec_i64 v, usize index)
{
	i64 result = 0;
	result = v.data[index];
	return result;
}

inline f32 vec_f32_at(vec_f32 v, usize index)
{
	f32 result = 0;
	result = v.data[index];
	return result;
}

inline f64 vec_f64_at(vec_f64 v, usize index)
{
	f64 result = 0;
	result = v.data[index];
	return result;
}
