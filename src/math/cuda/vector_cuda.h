#pragma once

#include "device.h"
#include "vector.h"
#include "util/types.h"

template <typename T>
void vec_to(vec<T>* v, Device device);

template <typename T>
vec<T> vec_zeros_gpu(size elements);

template <typename T>
vec<T> vec_full_gpu(size elements, T fill_value);

vec<f32> vec_rand_uniform_gpu(f32 min, f32 max, size n);
vec<f32> vec_rand_gauss_gpu(f32 mean, f32 std_dev, size n);
vec<f32> vec_rand_gauss_standard_gpu(size n);

template <typename T>
vec<T> vec_copy_gpu(vec<T> v);

template <typename T>
void vec_set_range_gpu(vec<T> v, vec<T> data, size offset);

template <typename T>
void vec_add_gpu(vec<T> a, vec<T> b);

template <typename T>
void vec_scale_gpu(vec<T> v, T c);

template <typename T>
vec<T> vec_scale_inv_gpu(vec<T> v, T c);

template <typename T>
void vec_had_gpu(vec<T> a, vec<T> b);

template <typename T>
T vec_sum_gpu(vec<T> v);
