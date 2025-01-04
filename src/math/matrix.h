#pragma once

#include "device.h"
#include "types.h"
#include "vector.h"

#include <stdio.h>
#include <stdlib.h>

enum Axis : u8
{
    Axis_Rows = 0,
    Axis_Cols
};

template <typename T>
struct mat
{
    static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                  "Type must be integral or floating-point.");
    usize rows;
    usize cols;
    T* data;
    Device device;
};

template <typename T>
internal mat<T> mat_init(usize rows, usize cols, T* data, Device device=Device_CPU);

template <typename T>
internal mat<T> mat_zeros(usize rows, usize cols, Device device=Device_CPU);

template <typename T>
internal mat<T> mat_full(usize rows, usize cols, T fill_value, Device device=Device_CPU);

mat<f32> mat_rand_uniform(f32 min, f32 max, usize rows, usize cols);
mat<f32> mat_rand_gauss(f32 mean, f32 std_dev, usize rows, usize cols);
mat<f32> mat_rand_gauss_standard(usize rows, usize cols);

// Set values of a matrix row with same-size vector
template <typename T>
internal void mat_set_row(mat<T> m, vec<T> data, usize row);

// Set values of a matrix column with same-size vector
template <typename T>
internal void mat_set_col(mat<T> m, vec<T> data, usize col);

template<typename T>
internal vec<T> mat_get_row(mat<T> m, usize row);

template <typename T>
internal vec<T> mat_get_col(mat<T> m, usize col);

// Set a range of values within a row of a matrix starting at row_offset
template <typename T>
internal void mat_set_row_range(mat<T> m, vec<T> data, usize row, usize row_offset);

// Set a range of values within a column of a matrix starting at col_offset
template <typename T>
internal void mat_set_col_range(mat<T> m, vec<T> data, usize col, usize col_offset);

template <typename T>
internal void mat_print(mat<T> m);

template <typename T>
internal mat<T> mat_stretch_cols(mat<T> orig, mat<T> target);

template <typename T>
internal mat<T> mat_stretch_rows(mat<T> orig, mat<T> target);

template <typename T>
internal void mat_add(mat<T> a, mat<T> b);

template <typename T>
internal mat<T> mat_stretch_add(mat<T> a, mat<T> b);

template <typename T>
internal void mat_scale(mat<T> m, T scale);

/* Scale a matrix by a vector along either axis
 * such that scaling along the rows will scale the ith value of each row with the ith vector element,
 * and scaling along the columns will scale the ith value of each column with the ith vector element.
 */
template <typename T>
internal void mat_scale(mat<T> m, vec<T> scale, Axis axis=Axis_Rows);

template <typename T>
internal mat<T> mat_mul(mat<T> a, mat<T> b);

/* Element-wise (Hadamard) product
 * Matrices must be equal size
 */
template <typename T>
internal void mat_had(mat<T> a, mat<T> b);

// Sum each row or column of a matrix and return the result as a vector.
template <typename T>
internal vec<T> mat_sum(mat<T> m, Axis axis=Axis_Rows);

template <typename T>
internal inline T mat_at(mat<T> m, usize row, usize col)
{
    return m.data[m.cols*row + col];
}

template <typename T>
internal inline void mat_set_val(mat<T> m, usize row, usize col, T val)
{
    m.data[m.cols*row + col] = val;
}
