#pragma once

#include "util/types.h"

#include "matrix.cpp"
#include "vector.cpp"

#include <math.h>

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

// Generate a sequence of n evenly-spaced numbers in the range [x1, x2]
vec<f32> linspace(f32 x1, f32 x2, size n, Device device=Device_CPU);
vec<f64> linspace(f64 x1, f64 x2, size n, Device device=Device_CPU);

internal inline f32 sin_f32(f32 x)
{
    f32 result = sinf(x);
    return result;
}

internal inline f64 sin_f64(f64 x)
{
    f64 result = sin(x);
    return result;
}

internal inline f32 cos_f32(f32 x)
{
    f32 result = cosf(x);
    return result;
}

internal inline f64 cos_f64(f64 x)
{
    f64 result = cos(x);
    return result;
}

vec<f32> sin_vec(vec<f32> v);
vec<f64> sin_vec(vec<f64> v);

vec<f32> cos_vec(vec<f32> v);
vec<f64> cos_vec(vec<f64> v);

void exp_vec(vec<f32> v);
void exp_vec(vec<f64> v);

void exp_mat(mat<f32> m);
void exp_mat(mat<f64> m);
