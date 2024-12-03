#pragma once

#include "device.h"
#include "types.h"
#include "vector.h"

template <typename T>
void vec_to(vec<T>* v, Device device);

template <typename T>
vec<T> vec_zeros_gpu(usize elements);

template <typename T>
vec<T> vec_full_gpu(usize elements, T fill_value);

template <typename T>
void vec_set_range_gpu(vec<T> v, vec<T> data, usize offset);

template <typename T>
void vec_add_gpu(vec<T> a, vec<T> b);

template <typename T>
void vec_scale_gpu(vec<T> v, T c);

template <typename T>
vec<T> vec_reciprocal_gpu(vec<T> v);

template <typename T>
void vec_had_gpu(vec<T> a, vec<T> b);

f32 vec_sum_gpu(vec<f32> v);
f64 vec_sum_gpu(vec<f64> v);
u32 vec_sum_gpu(vec<u32> v);
i32 vec_sum_gpu(vec<i32> v);
