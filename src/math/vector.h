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
    size elements;
    T* data;
    Device device;

    inline T& operator[](size i)
    {
        ASSERT(i < elements);
        return data[i];
    }
};

// In-place addition
template <typename T>
vec<T> operator+=(vec<T> a, vec<T> b)
{
    ASSERT(a.elements == b.elements);
    ASSERT(a.device == b.device);

    switch (a.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < a.elements; ++i)
                a[i] += b[i];
        } break;

        case Device_GPU: vec_add_gpu(a, b); break;

        default: break;
    }

    return a;
}

// In-place vector-scalar multiplication
template <typename T>
vec<T> operator*=(vec<T> v, T c)
{
    switch (v.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < v.elements; ++i)
                v[i] *= c;
        } break;

        case Device_GPU: vec_scale_gpu(v, c); break;

        default: break;
    }

    return v;
}

// In-place vector-vector multiplication (Hadamard product)
template <typename T>
vec<T> operator *=(vec<T> a, vec<T> b)
{
    vec_had(a, b);
    return a;
}

template <typename T>
internal vec<T> vec_init(size elements, T* data, Device device=Device_CPU);

template<typename T>
vec<T> vec_zeros(size elements, Device device=Device_CPU);

// Create vector of a given size filled with fill_value
template <typename T>
internal vec<T> vec_full(size elements, T value, Device device=Device_CPU);

vec<f32> vec_rand_uniform(f32 min, f32 max, size n);
vec<f32> vec_rand_gauss(f32 mean, f32 std_dev, size n);
vec<f32> vec_rand_gauss_standard(size n);

// Set multiple vector values beginning at offset
template <typename T>
internal void vec_set_range(vec<T> v, vec<T> data, size offset);

template <typename T>
internal void vec_print(vec<T> v);

// Take the reciprocal of each element of the vector.
template <typename T>
internal vec<T> vec_reciprocal(vec<T> v);

// Element-wise (Hadamard product)
// Vectors must be equal size
template <typename T>
internal vec<T> vec_had(vec<T> a, vec<T> b);

// Sum all elements of a vector.
template <typename T>
internal T vec_sum(vec<T> v);
