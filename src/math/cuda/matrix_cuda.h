#pragma once

#include "device.h"
#include "types.h"
#include "matrix.h"

template <typename T>
void mat_to(mat<T>* m, Device device);

template <typename T>
void mat_free_data_gpu(mat<T> m);

template <typename T>
mat<T> mat_zeros_gpu(size rows, size cols);

template <typename T>
mat<T> mat_full_gpu(size rows, size cols, T fill_value);

mat<f32> mat_rand_uniform_gpu(size rows, size cols, f32 min, f32 max);
mat<f32> mat_rand_gauss_gpu(size rows, size cols, f32 mean, f32 std_dev);
mat<f32> mat_rand_gauss_standard_gpu(size rows, size cols);

template <typename T>
mat<T> mat_copy_gpu(mat<T> m);

template <typename T>
void mat_set_row_gpu(mat<T> m, vec<T> data, size row);

template <typename T>
void mat_set_col_gpu(mat<T> m, vec<T> data, size col);

template <typename T>
void mat_set_row_range_gpu(mat<T> m, vec<T> data, size row, size row_offset);

template <typename T>
void mat_set_col_range_gpu(mat<T> m, vec<T> data, size col, size col_offset);

template <typename T>
mat<T> mat_stretch_cols_gpu(mat<T> orig, mat<T> target);

template <typename T>
mat<T> mat_stretch_rows_gpu(mat<T> orig, mat<T> target);

template <typename T>
void mat_add_gpu(mat<T> a, mat<T> b);

template <typename T>
void mat_scale_gpu(mat<T> m, T scale);

template <typename T>
void mat_scale_gpu(mat<T> m, vec<T> scale, Axis axis);

template <typename T>
mat<T> mat_mul_gpu(mat<T> a, mat<T> b);

template <typename T>
void mat_had_gpu(mat<T> a, mat<T> b);

template <typename T>
vec<T> mat_sum_gpu(mat<T>m, Axis axis);
