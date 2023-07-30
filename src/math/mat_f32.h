#pragma once

#include "util/types.h"

#include <stdio.h>
#include <stdlib.h>

typedef struct mat_f32
{
    usize rows;
    usize cols;
    f32* data;
} mat_f32;

mat_f32 mat_f32_init(usize rows, usize cols, f32* data);
mat_f32 mat_f32_zeros(usize rows, usize cols);

f32 mat_f32_at(mat_f32 m, usize row, usize col);                 // Return value of matrix at row,col
void mat_f32_set_val(mat_f32* m, usize row, usize col, f32 val); // Set value of matrix at row,col

void mat_f32_print(mat_f32 m);

mat_f32 mat_f32_add(mat_f32 a, mat_f32 b);
mat_f32 mat_f32_mul(mat_f32 a, mat_f32 b);
