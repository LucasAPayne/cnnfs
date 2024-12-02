#pragma once

#include "device.h"
#include "types.h"
#include "matrix.h"

template <typename T>
void mat_to(mat<T>* m, Device device);

template <typename T>
void mat_free_data_gpu(mat<T> m);

template <typename T>
mat<T> mat_zeros_gpu(usize rows, usize cols);

template <typename T>
mat<T> mat_full_gpu(usize rows, usize cols, T fill_value);

template <typename T>
void mat_set_row_gpu(mat<T> m, vec<T> data, usize row);

template <typename T>
void mat_set_col_gpu(mat<T> m, vec<T> data, usize col);

template <typename T>
void mat_set_row_range_gpu(mat<T> m, vec<T> data, usize row, usize row_offset);

template <typename T>
void mat_set_col_range_gpu(mat<T> m, vec<T> data, usize col, usize col_offset);

template <typename T>
mat<T> mat_stretch_cols_gpu(mat<T> orig, mat<T> target);

template <typename T>
mat<T> mat_stretch_rows_gpu(mat<T> orig, mat<T> target);

template <typename T>
void mat_add_gpu(mat<T> a, mat<T> b);

template <typename T>
void mat_scale_gpu(mat<T> m, T value);

template <typename T>
mat<T> mat_mul_gpu(mat<T> a, mat<T> b);

template <typename T>
void mat_had_gpu(mat<T> a, mat<T> b);
