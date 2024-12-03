#pragma once

#include "device.h"
#include "types.h"

#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T>
struct vec
{
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "Type must be integral or floating-point.");
    usize elements;
    T* data;
    Device device;
};

template <typename T>
internal vec<T> vec_init(usize elements, T* data, Device device=DEVICE_CPU);

template<typename T>
vec<T> vec_zeros(usize elements, Device device=DEVICE_CPU);

// Create vector of a given size filled with fill_value
template <typename T>
internal vec<T> vec_full(usize elements, T value, Device device=DEVICE_CPU);

vec<f32> vec_rand_uniform(f32 min, f32 max, usize n);
vec<f32> vec_rand_gauss(f32 mean, f32 std_dev, usize n);
vec<f32> vec_rand_gauss_standard(usize n);

// Set multiple vector values beginning at offset
template <typename T>
internal void vec_set_range(vec<T> v, vec<T> data, usize offset);

template <typename T>
internal void vec_print(vec<T> v);

template <typename T>
internal void vec_add(vec<T> a, vec<T> b);

template <typename T>
internal void vec_scale(vec<T> v, T c);

// Take the reciprocal of each element of the vector.
template <typename T>
internal vec<T> vec_reciprocal(vec<T> v);

// Element-wise (Hadamard product)
// Vectors must be equal size
template <typename T>
internal void vec_had(vec<T> a, vec<T> b);

// Sum all elements of a vector.
template <typename T>
internal T vec_sum(vec<T> v);

template <typename T>
internal inline T vec_at(vec<T> v, usize index)
{
    T result = v.data[index];
    return result;
}
