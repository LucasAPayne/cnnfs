#include "mat_f32.h"
#include "vector/vec_f32.h"

#include <stdio.h>
#include <stdlib.h>

mat_f32 mat_f32_init(usize rows, usize cols, f32* data)
{
    mat_f32 result = {0};
    result.rows = rows;
    result.cols = cols;
    result.data = data;

    return result;
}

mat_f32 mat_f32_zeros(usize rows, usize cols)
{
    mat_f32 result = {0};
    result.rows = rows;
    result.cols = cols;

    result.data = calloc(rows*cols, sizeof(f32));
    ASSERT(result.data);

    return result;
}

mat_f32 mat_f32_full(usize rows, usize cols, f32 fill_value)
{
    mat_f32 m = mat_f32_zeros(rows, cols);

    for (usize row = 0; row < m.rows; ++row)
    {
        for (usize col = 0; col < m.cols; ++col)
            mat_f32_set_val(&m, row, col, fill_value);
    }

    return m;
}

void mat_f32_set_row(mat_f32* m, vec_f32 v, usize row)
{
    // To set a row, the vector must have the same number of elements as
    // the matrix has columns
    ASSERT(m->cols == v.elements);

    for (usize col = 0; col < m->cols; ++col)
        mat_f32_set_val(m, row, col, v.data[col]);
}

void mat_f32_set_col(mat_f32* m, vec_f32 v, usize col)
{
    // To set a column, the vector must have the same number of elemnts as
    // the matrix has rows
    ASSERT(m->rows == v.elements);

    for (usize row = 0; row < m->rows; ++row)
        mat_f32_set_val(m, row, col, v.data[row]);
}

void mat_f32_set_row_range(mat_f32* m, vec_f32 v, usize row, usize row_offset)
{
    // There must be enough columns after the offset to accommodate the vector
    ASSERT(m->cols >= v.elements + row_offset);

    for (usize i = 0; i < v.elements; ++i)
        mat_f32_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_f32_set_col_range(mat_f32* m, vec_f32 v, usize col, usize col_offset)
{
    // There must be enough rows after the offset to accomodate the vector
    ASSERT(m->rows >= v.elements + col_offset);

    for (usize i = 0; i < v.elements; ++i)
        mat_f32_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_f32_print(mat_f32 m)
{
    int width = 0;

    // Compute max width needed
    for (usize i = 0; i < m.rows; ++i)
    {
        for (usize j = 0; j < m.cols; ++j)
        {
            int w = snprintf(NULL, 0, "%f", mat_f32_at(m, i, j));
            if (width < w)
                width = w;
        }
    }

    // Print
    for (usize i = 0; i < m.rows; ++i)
    {
        printf("[");
        for (usize j = 0; j < m.cols; ++j)
        {
            if (j != 0) printf(", ");
            printf("%*f", width, mat_f32_at(m, i, j));
        }
        printf("]\n");
    }
}

// Stretch a row vector to be matrix with the same shape as target
// by copying the original row a number of times
internal mat_f32 mat_stretch_cols(mat_f32 orig, mat_f32 target)
{
    // The original matrix must have the same number of columns as the target matrix
    ASSERT(orig.rows == target.rows);
    mat_f32 result = mat_f32_zeros(target.rows, target.cols);
    for (usize i = 0; i < target.rows; ++i)
    {
        for (usize j = 0; j < target.cols; ++j)
        {
            f32 val = mat_f32_at(orig, i, 0);
            mat_f32_set_val(&result, i, j, val);
        }
    }

    return result;
}

// Stretch a column vector to be matrix with the same shape as target
// by copying the original column a number of times
internal mat_f32 mat_stretch_rows(mat_f32 orig, mat_f32 target)
{
    // The original matrix must have the same number of rows as the target matrix
    ASSERT(orig.cols == target.cols);
    mat_f32 result = mat_f32_zeros(target.rows, target.cols);
    for (usize i = 0; i < target.rows; ++i)
    {
        for (usize j = 0; j < target.cols; ++j)
        {
            f32 val = mat_f32_at(orig, 0, j);
            mat_f32_set_val(&result, i, j, val);
        }
    }
    return result;
}

mat_f32 mat_f32_add(mat_f32 a, mat_f32 b)
{
    /* NOTE(lucas): Matrices must be the same size in both dimensions,
     * or must be the same size in one dimension while one matrix is a row/column vector.
     * In the latter case, add the row/column vector across the matrix
     */
    b32 a_col_vec = a.cols == 1;
    b32 a_row_vec = a.rows == 1;
    b32 b_col_vec = b.cols == 1;
    b32 b_row_vec = b.rows == 1;
    b32 valid_sizes = ((a.rows == b.rows)       && (a.cols == b.cols))
                   || ((a_col_vec || b_col_vec) && (a.rows == b.rows))
                   || ((a_row_vec || b_row_vec) && (a.cols == b.cols));
    ASSERT(valid_sizes);

    // NOTE(lucas): If one matrix is a row/column vector and the other is not,
    // construct a new matrix of appropriate size by copying rows/columns
    if ((a.rows != b.rows) || (a.cols != b.cols))
    {
        if (b_row_vec)
            b = mat_stretch_rows(b, a);
        else if (b_col_vec)
            b = mat_stretch_cols(b, a);
        else if (a_row_vec)
            a = mat_stretch_cols(a, b);
        else if (a_col_vec)
            a = mat_stretch_cols(a, b);
    }

    mat_f32 result = mat_f32_zeros(a.rows, a.cols);

    for (usize i = 0; i < result.rows; ++i)
    {
        for (usize j = 0; j < result.cols; ++j)
        {
            f32 val = mat_f32_at(a, i, j) + mat_f32_at(b, i, j);
            mat_f32_set_val(&result, i, j, val);
        }
    }

    return result;
}

mat_f32 mat_f32_mul(mat_f32 a, mat_f32 b)
{
    // For multiplication to be valid, the number of columns in A must equal the number of rows in B
    ASSERT(a.cols == b.rows);

    // Shape of output matrix is determined by the number of rows in A and the number of columns in B
    mat_f32 result = mat_f32_zeros(a.rows, b.cols);

    // TODO(lucas): Replace this. This is the purely naive way to multiply matrices.
    for (usize i = 0; i < result.rows; ++i)
    {
        for (usize j = 0; j < result.cols; ++j)
        {
            f32 sum = 0;
            for (usize k = 0; k < a.cols; ++k)
                sum += mat_f32_at(a, i, k) * mat_f32_at(b, k, j);
            
            mat_f32_set_val(&result, i, j, sum);
        }
    }
    return result;
}

mat_f32 mat_f32_had(mat_f32 a, mat_f32 b)
{
    ASSERT(a.rows == b.rows);
    ASSERT(a.cols == b.cols);

    mat_f32 result = mat_f32_zeros(a.rows, a.cols);

    for (usize row = 0; row < result.rows; ++row)
    {
        for (usize col = 0; col < result.cols; ++col)
        {
            f32 val = mat_f32_at(a, row, col) * mat_f32_at(b, row, col);
            mat_f32_set_val(&result, row, col, val);
        }
    }

    return result;
}
