#pragma once

#include "util/types.h"

typedef struct mat_u8
{
    usize rows;
    usize cols;
    u8* data;
} mat_u8;

mat_u8 mat_u8_init(usize rows, usize cols, u8* data);
mat_u8 mat_u8_zeros(usize rows, usize cols);

u8 mat_u8_at(mat_u8 m, usize row, usize col);                 // Return value of matrix at row,col
void mat_u8_set_val(mat_u8* m, usize row, usize col, u8 val); // Set value of matrix at row,col

void mat_u8_print(mat_u8 m);

mat_u8 mat_u8_add(mat_u8 a, mat_u8 b);
mat_u8 mat_u8_mul(mat_u8 a, mat_u8 b);
