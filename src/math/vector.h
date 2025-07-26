#pragma once

#include "device.h"
#include "types.h"

#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include "log.h"
#define log_invalid_device(device) \
        log_error("[" LOG_FILE_PATH ":" LOG_LINE_NUM "] in " LOG_FUNC_NAME ": Invalid device: %d. " \
        "Device must be Device_CPU (0) or Device_GPU (1).\n", __FILE__, __LINE__, __func__, device)

#define log_invalid_axis(axis) \
        log_error("[" LOG_FILE_PATH ":" LOG_LINE_NUM "] in " LOG_FUNC_NAME ": Invalid axis: %d. " \
        "Axis must be Axis_Rows (0) or Axis_Cols (1).\n", __FILE__, __LINE__, __func__, axis)

template <typename T>
struct vec
{
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "Type must be integral or floating-point.");
    T* data;
    size elements;
    Device device;

    inline T& operator[](size i)
    {
        ASSERTF(i < elements, "Index out of range (max: %llu, got: %llu).", elements-1, i);
        return data[i];
    }
};

// Addition with copy, does not modify parameters
template <typename T>
vec<T> operator+(vec<T> a, vec<T> b)
{
    vec<T> result = vec_add(a, b, false);
    return result;
}

// In-place addition
template <typename T>
vec<T> operator+=(vec<T> a, vec<T> b)
{
    vec<T> result = vec_add(a, b);
    return result;
}

// vector-scalar multiplation (scale) with copy, does not modify parameters
template <typename T>
vec<T> operator*(vec<T> v, T c)
{
    vec<T> result = vec_scale(v, c, false);
    return result;
}

// In-place vector-scalar multiplication (scale)
template <typename T>
vec<T> operator*=(vec<T> v, T c)
{
    vec<T> result = vec_scale(v, c);
    return result;
}

// scalar-vector multiplication (scale) with copy, does not modify parameters
template <typename T>
vec<T> operator*(T c, vec<T> v)
{
    vec<T> result = vec_scale(v, c, false);
    return result;
}

// In-place scalar-vector multiplication (scale)
template <typename T>
vec<T> operator*=(T c, vec<T> v)
{
    vec<T> result = vec_scale(v, c);
    return result;
}

// In-place vector-vector multiplication (Hadamard product)
template <typename T>
vec<T> operator *=(vec<T> a, vec<T> b)
{
    vec<T> result = vec_had(a, b);
    return result;
}

// In-place vector-scalar division (scale)
template <typename T>
vec<T> operator/=(vec<T> v, T c)
{
    T scale = (T)1/c;
    vec<T> result = vec_scale(v, scale);
    return result;
}

// vector-scalar division (scale) with copy, does not modify parameters
template <typename T>
vec<T> operator/(vec<T> v, T c)
{
    T scale = (T)1/c;
    vec<T> result = vec_scale(v, scale, false);
    return result;
}

// scalar-vector division (inverse scale) with copy, does not modify parameters
template <typename T>
vec<T> operator/(T c, vec<T> v)
{
    vec<T> result = vec_scale_inv(v, c, false);
    return result;
}

template <typename T>
internal vec<T> vec_init(size elements, T* data, Device device=Device_CPU);

template<typename T>
vec<T> vec_zeros(size elements, Device device=Device_CPU);

// Create vector of a given size filled with fill_value
template <typename T>
internal vec<T> vec_full(size elements, T value, Device device=Device_CPU);

internal vec<f32> vec_rand_uniform(size n, f32 min, f32 max, Device device=Device_CPU);
internal vec<f32> vec_rand_gauss(size n, f32 mean, f32 std_dev, Device device=Device_CPU);
internal vec<f32> vec_rand_gauss_standard(size n, Device device=Device_CPU);

// Create a new vector and copy in values from another vector
template <typename T>
internal vec<T> vec_copy(vec<T> v);

// Set multiple vector values beginning at offset
template <typename T>
internal void vec_set_range(vec<T> v, vec<T> data, size offset);

template <typename T>
internal void vec_print(vec<T> v);

template <typename T>
internal vec<T> vec_add(vec<T> a, vec<T> b, b32 in_place=true);

template <typename T>
internal vec<T> vec_scale(vec<T> a, T c, b32 in_place=true);

// Calculates c/v[i] for each value v[i] in v.
template <typename T>
internal vec<T> vec_scale_inv(vec<T> v, T c, b32 in_place=true);

// Take the reciprocal of each element of the vector.
template <typename T>
internal vec<T> vec_reciprocal(vec<T> v, b32 in_place=true);

// Element-wise (Hadamard product)
// Vectors must be equal size
template <typename T>
internal vec<T> vec_had(vec<T> a, vec<T> b, b32 in_place=true);

// Sum all elements of a vector.
template <typename T>
internal T vec_sum(vec<T> v);
