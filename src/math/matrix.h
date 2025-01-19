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
    size rows;
    size cols;
    T* data;
    Device device;

    /* NOTE(lucas): The () operator is used over the [] operator because [] only accepts a single value.
     * To implement this, a matrix would have to be made up of row vectors, or one would have to be constructed
     * so that another [] operator can be chained on the result.
     * Therefore, it is simply more efficient and convenient to use the () operator instead.
     */
    inline T& operator()(size row, size col)
    {
        ASSERT(row < rows);
        ASSERT(col < cols);
        return data[row*cols + col];
    }
};

// Addition with copy, does not modify parameters
template <typename T>
mat<T> operator+(mat<T> a, mat<T> b)
{
    mat<T> result = mat_add(a, b, false);
    return result;
}

// In-place addition
template <typename T>
mat<T> operator+=(mat<T> a, mat<T> b)
{
    mat<T> result = mat_add(a, b);
    return result;
}

// TODO(lucas): matrix-vector/vector-matrix multiplication operators?

// matrix-scalar multiplation (scale) with copy, does not modify parameters
template <typename T>
mat<T> operator*(mat<T> v, T c)
{
    mat<T> result = mat_scale(v, c, false);
    return result;
}

// In-place matrix-scalar multiplication (scale)
template <typename T>
mat<T> operator*=(mat<T> v, T c)
{
    mat<T> result = mat_scale(v, c);
    return result;
}

// scalar-matrix multiplication (scale) with copy, does not modify parameters
template <typename T>
mat<T> operator*(T c, mat<T> v)
{
    mat<T> result = mat_scale(v, c, false);
    return result;
}

// In-place scalar-matrix multiplication (scale)
template <typename T>
mat<T> operator*=(T c, mat<T> v)
{
    mat<T> result = vec_mat(v, c);
    return result;
}

// In-place matrix-matrix multiplication (Hadamard product)
template <typename T>
mat<T> operator *=(mat<T> a, mat<T> b)
{
    mat<T> result = mat_had(a, b);
    return result;
}

template <typename T>
internal mat<T> mat_init(size rows, size cols, T* data, Device device=Device_CPU);

template <typename T>
internal mat<T> mat_zeros(size rows, size cols, Device device=Device_CPU);

template <typename T>
internal mat<T> mat_full(size rows, size cols, T fill_value, Device device=Device_CPU);

template <typename T>
internal mat<T> mat_copy(mat<T> m);

mat<f32> mat_rand_uniform(f32 min, f32 max, size rows, size cols);
mat<f32> mat_rand_gauss(f32 mean, f32 std_dev, size rows, size cols);
mat<f32> mat_rand_gauss_standard(size rows, size cols);

// Set values of a matrix row with same-size vector
template <typename T>
internal void mat_set_row(mat<T> m, vec<T> data, size row);

// Set values of a matrix column with same-size vector
template <typename T>
internal void mat_set_col(mat<T> m, vec<T> data, size col);

template<typename T>
internal vec<T> mat_get_row(mat<T> m, size row);

template <typename T>
internal vec<T> mat_get_col(mat<T> m, size col);

// Set a range of values within a row of a matrix starting at row_offset
template <typename T>
internal void mat_set_row_range(mat<T> m, vec<T> data, size row, size row_offset);

// Set a range of values within a column of a matrix starting at col_offset
template <typename T>
internal void mat_set_col_range(mat<T> m, vec<T> data, size col, size col_offset);

template <typename T>
internal void mat_print(mat<T> m);

template <typename T>
internal mat<T> mat_stretch_cols(mat<T> orig, mat<T> target);

template <typename T>
internal mat<T> mat_stretch_rows(mat<T> orig, mat<T> target);

template <typename T>
internal mat<T> mat_add(mat<T> a, mat<T> b, b32 in_place=true);

template <typename T>
internal mat<T> mat_stretch_add(mat<T> a, mat<T> b);

template <typename T>
internal mat<T> mat_scale(mat<T> m, T scale, b32 in_place=true);

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
internal mat<T> mat_had(mat<T> a, mat<T> b, b32 in_place=true);

// Sum each row or column of a matrix and return the result as a vector.
template <typename T>
internal vec<T> mat_sum(mat<T> m, Axis axis=Axis_Rows);
