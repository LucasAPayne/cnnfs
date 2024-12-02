#pragma once

#include "device.h"
#include "types.h"
#include "vector.h"

vec<f32> linspace_gpu(f32 x1, f32 x2, usize n);
vec<f64> linspace_gpu(f64 x1, f64 x2, usize n);

vec<f32> sin_vec_gpu(vec<f32> v);
vec<f64> sin_vec_gpu(vec<f64> v);

vec<f32> cos_vec_gpu(vec<f32> v);
vec<f64> cos_vec_gpu(vec<f64> v);
