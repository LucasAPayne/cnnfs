#pragma once

#include "util/types.h"
#include "vector/vec_u8.h"

typedef struct mat_u8
{
    usize rows;
    usize cols;
    u8* data;
} mat_u8;

mat_u8 mat_u8_init(usize rows, usize cols, u8* data);
mat_u8 mat_u8_zeros(usize rows, usize cols);

// Create matrix of a given size filled with fill_value
mat_u8 mat_u8_full(usize rows, usize cols, u8 fill_value);

// Return value of matrix at row,col
inline u8 mat_u8_at(mat_u8 m, usize row, usize col)
{
    return m.data[m.cols*row + col];
}

// Set value of matrix at row,col
inline void mat_u8_set_val(mat_u8* m, usize row, usize col, u8 val)
{
    m->data[m->cols*row + col] = val;
}
// Set values of a matrix row or column with same-size vector
void mat_u8_set_row(mat_u8* m, vec_u8 v, usize row);
void mat_u8_set_col(mat_u8* m, vec_u8 v, usize col);

// Set a range of values within a row or column of a matrix,
// but not necessarily the entire row.
void mat_u8_set_row_range(mat_u8* m, vec_u8 v, usize row, usize row_offset);
void mat_u8_set_col_range(mat_u8* m, vec_u8 v, usize col, usize col_offset);

void mat_u8_print(mat_u8 m);

mat_u8 mat_u8_add(mat_u8 a, mat_u8 b);
mat_u8 mat_u8_mul(mat_u8 a, mat_u8 b);

// Element-wise (Hadamard product)
// Matrices must be equal size
mat_u8 mat_u8_had(mat_u8 a, mat_u8 b);
