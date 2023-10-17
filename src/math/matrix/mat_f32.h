#pragma once

#include "util/types.h"
#include "vector/vec_f32.h"

typedef struct mat_f32
{
    usize rows;
    usize cols;
    f32* data;
} mat_f32;

mat_f32 mat_f32_init(usize rows, usize cols, f32* data);
mat_f32 mat_f32_zeros(usize rows, usize cols);

// Return value of matrix at row,col
inline f32 mat_f32_at(mat_f32 m, usize row, usize col)
{
    return m.data[m.cols*row + col];
}

// Set value of matrix at row,col
inline void mat_f32_set_val(mat_f32* m, usize row, usize col, f32 val)
{
    m->data[m->cols*row + col] = val;
}

// Set values of a matrix row or column with same-size vector
void mat_f32_set_row(mat_f32* m, vec_f32 v, usize row);
void mat_f32_set_col(mat_f32* m, vec_f32 v, usize col);

// Set a range of values within a row or column of a matrix,
// but not necessarily the entire row.
void mat_f32_set_row_range(mat_f32* m, vec_f32 v, usize row, usize row_offset);
void mat_f32_set_col_range(mat_f32* m, vec_f32 v, usize col, usize col_offset);

void mat_f32_print(mat_f32 m);

mat_f32 mat_f32_scale(mat_f32 m, f32 value);

mat_f32 mat_f32_add(mat_f32 a, mat_f32 b);
mat_f32 mat_f32_mul(mat_f32 a, mat_f32 b);

// Element-wise (Hadamard product)
// Matrices must be equal size
mat_f32 mat_f32_had(mat_f32 a, mat_f32 b);
