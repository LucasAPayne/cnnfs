#pragma once

#include "device.h"
#include "types.h"
#include "vector.h"

template <typename T>
void vec_to(vec<T>* v, Device device);

template <typename T>
vec<T> vec_zeros_gpu(size elements);

template <typename T>
vec<T> vec_full_gpu(size elements, T fill_value);

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
