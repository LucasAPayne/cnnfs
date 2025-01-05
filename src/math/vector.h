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

    T& operator[](size i)
    {
        ASSERT(i < elements);
        return data[i];
    }

    vec& operator+=(vec<T>& other)
    {
        ASSERT(elements == other.elements);
        ASSERT(device == other.device);

        switch (device)
        {
            case Device_CPU:
            {
                for (size i = 0; i < elements; ++i)
                    data[i] += other[i];
            } break;

            case Device_GPU: vec_add_gpu(*this, other); break;

            default: break;
        }
        return *this;
    }
};

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
internal inline T vec_at(vec<T> v, size index)
{
    T result = v[index];
    return result;
}
