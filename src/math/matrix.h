#pragma once

#include "../util/types.h"
#include "matrix_cuda.h"

#include "vector.h"

#include <stdio.h>

#include <stdlib.h>

typedef enum Device
{
	DEVICE_CPU = 0,
	DEVICE_GPU
} Device;


typedef struct mat_u8
{
	usize rows;
	usize cols;
	u8* data;
} mat_u8;

typedef struct mat_u16
{
	usize rows;
	usize cols;
	u16* data;
} mat_u16;

typedef struct mat_u32
{
	usize rows;
	usize cols;
	u32* data;
} mat_u32;

typedef struct mat_u64
{
	usize rows;
	usize cols;
	u64* data;
} mat_u64;

typedef struct mat_i8
{
	usize rows;
	usize cols;
	i8* data;
} mat_i8;

typedef struct mat_i16
{
	usize rows;
	usize cols;
	i16* data;
} mat_i16;

typedef struct mat_i32
{
	usize rows;
	usize cols;
	i32* data;
} mat_i32;

typedef struct mat_i64
{
	usize rows;
	usize cols;
	i64* data;
} mat_i64;

typedef struct mat_f32
{
	Device device;
	usize rows;
	usize cols;
	f32* data;
} mat_f32;

typedef struct mat_f64
{
	usize rows;
	usize cols;
	f64* data;
} mat_f64;

mat_u8 mat_u8_init(usize rows, usize cols, u8* data);
mat_u16 mat_u16_init(usize rows, usize cols, u16* data);
mat_u32 mat_u32_init(usize rows, usize cols, u32* data);
mat_u64 mat_u64_init(usize rows, usize cols, u64* data);
mat_i8 mat_i8_init(usize rows, usize cols, i8* data);
mat_i16 mat_i16_init(usize rows, usize cols, i16* data);
mat_i32 mat_i32_init(usize rows, usize cols, i32* data);
mat_i64 mat_i64_init(usize rows, usize cols, i64* data);
mat_f32 mat_f32_init(usize rows, usize cols, f32* data, Device device);
mat_f64 mat_f64_init(usize rows, usize cols, f64* data);

mat_u8 mat_u8_zeros(usize rows, usize cols);
mat_u16 mat_u16_zeros(usize rows, usize cols);
mat_u32 mat_u32_zeros(usize rows, usize cols);
mat_u64 mat_u64_zeros(usize rows, usize cols);
mat_i8 mat_i8_zeros(usize rows, usize cols);
mat_i16 mat_i16_zeros(usize rows, usize cols);
mat_i32 mat_i32_zeros(usize rows, usize cols);
mat_i64 mat_i64_zeros(usize rows, usize cols);
mat_f32 mat_f32_zeros(usize rows, usize cols);
mat_f64 mat_f64_zeros(usize rows, usize cols);

void mat_u8_set_row(mat_u8* m, vec_u8 v, usize row);
void mat_u16_set_row(mat_u16* m, vec_u16 v, usize row);
void mat_u32_set_row(mat_u32* m, vec_u32 v, usize row);
void mat_u64_set_row(mat_u64* m, vec_u64 v, usize row);
void mat_i8_set_row(mat_i8* m, vec_i8 v, usize row);
void mat_i16_set_row(mat_i16* m, vec_i16 v, usize row);
void mat_i32_set_row(mat_i32* m, vec_i32 v, usize row);
void mat_i64_set_row(mat_i64* m, vec_i64 v, usize row);
void mat_f32_set_row(mat_f32* m, vec_f32 v, usize row);
void mat_f64_set_row(mat_f64* m, vec_f64 v, usize row);

void mat_u8_set_col(mat_u8* m, vec_u8 v, usize col);
void mat_u16_set_col(mat_u16* m, vec_u16 v, usize col);
void mat_u32_set_col(mat_u32* m, vec_u32 v, usize col);
void mat_u64_set_col(mat_u64* m, vec_u64 v, usize col);
void mat_i8_set_col(mat_i8* m, vec_i8 v, usize col);
void mat_i16_set_col(mat_i16* m, vec_i16 v, usize col);
void mat_i32_set_col(mat_i32* m, vec_i32 v, usize col);
void mat_i64_set_col(mat_i64* m, vec_i64 v, usize col);
void mat_f32_set_col(mat_f32* m, vec_f32 v, usize col);
void mat_f64_set_col(mat_f64* m, vec_f64 v, usize col);

void mat_u8_set_row_range(mat_u8* m, vec_u8 v, usize row, usize row_offset);
void mat_u16_set_row_range(mat_u16* m, vec_u16 v, usize row, usize row_offset);
void mat_u32_set_row_range(mat_u32* m, vec_u32 v, usize row, usize row_offset);
void mat_u64_set_row_range(mat_u64* m, vec_u64 v, usize row, usize row_offset);
void mat_i8_set_row_range(mat_i8* m, vec_i8 v, usize row, usize row_offset);
void mat_i16_set_row_range(mat_i16* m, vec_i16 v, usize row, usize row_offset);
void mat_i32_set_row_range(mat_i32* m, vec_i32 v, usize row, usize row_offset);
void mat_i64_set_row_range(mat_i64* m, vec_i64 v, usize row, usize row_offset);
void mat_f32_set_row_range(mat_f32* m, vec_f32 v, usize row, usize row_offset);
void mat_f64_set_row_range(mat_f64* m, vec_f64 v, usize row, usize row_offset);

void mat_u8_set_col_range(mat_u8* m, vec_u8 v, usize col, usize col_offset);
void mat_u16_set_col_range(mat_u16* m, vec_u16 v, usize col, usize col_offset);
void mat_u32_set_col_range(mat_u32* m, vec_u32 v, usize col, usize col_offset);
void mat_u64_set_col_range(mat_u64* m, vec_u64 v, usize col, usize col_offset);
void mat_i8_set_col_range(mat_i8* m, vec_i8 v, usize col, usize col_offset);
void mat_i16_set_col_range(mat_i16* m, vec_i16 v, usize col, usize col_offset);
void mat_i32_set_col_range(mat_i32* m, vec_i32 v, usize col, usize col_offset);
void mat_i64_set_col_range(mat_i64* m, vec_i64 v, usize col, usize col_offset);
void mat_f32_set_col_range(mat_f32* m, vec_f32 v, usize col, usize col_offset);
void mat_f64_set_col_range(mat_f64* m, vec_f64 v, usize col, usize col_offset);

void mat_u8_print(mat_u8 m);
void mat_u16_print(mat_u16 m);
void mat_u32_print(mat_u32 m);
void mat_u64_print(mat_u64 m);
void mat_i8_print(mat_i8 m);
void mat_i16_print(mat_i16 m);
void mat_i32_print(mat_i32 m);
void mat_i64_print(mat_i64 m);
void mat_f32_print(mat_f32 m);
void mat_f64_print(mat_f64 m);

mat_u8 mat_u8_scale(mat_u8 m, u8 value);
mat_u16 mat_u16_scale(mat_u16 m, u16 value);
mat_u32 mat_u32_scale(mat_u32 m, u32 value);
mat_u64 mat_u64_scale(mat_u64 m, u64 value);
mat_i8 mat_i8_scale(mat_i8 m, i8 value);
mat_i16 mat_i16_scale(mat_i16 m, i16 value);
mat_i32 mat_i32_scale(mat_i32 m, i32 value);
mat_i64 mat_i64_scale(mat_i64 m, i64 value);
mat_f32 mat_f32_scale(mat_f32 m, f32 value);
mat_f64 mat_f64_scale(mat_f64 m, f64 value);

mat_u8 mat_u8_add(mat_u8 a, mat_u8 b);
mat_u16 mat_u16_add(mat_u16 a, mat_u16 b);
mat_u32 mat_u32_add(mat_u32 a, mat_u32 b);
mat_u64 mat_u64_add(mat_u64 a, mat_u64 b);
mat_i8 mat_i8_add(mat_i8 a, mat_i8 b);
mat_i16 mat_i16_add(mat_i16 a, mat_i16 b);
mat_i32 mat_i32_add(mat_i32 a, mat_i32 b);
mat_i64 mat_i64_add(mat_i64 a, mat_i64 b);
mat_f32 mat_f32_add(mat_f32 a, mat_f32 b);
mat_f64 mat_f64_add(mat_f64 a, mat_f64 b);

mat_u8 mat_u8_mul(mat_u8 a, mat_u8 b);
mat_u16 mat_u16_mul(mat_u16 a, mat_u16 b);
mat_u32 mat_u32_mul(mat_u32 a, mat_u32 b);
mat_u64 mat_u64_mul(mat_u64 a, mat_u64 b);
mat_i8 mat_i8_mul(mat_i8 a, mat_i8 b);
mat_i16 mat_i16_mul(mat_i16 a, mat_i16 b);
mat_i32 mat_i32_mul(mat_i32 a, mat_i32 b);
mat_i64 mat_i64_mul(mat_i64 a, mat_i64 b);
mat_f32 mat_f32_mul(mat_f32 a, mat_f32 b);
mat_f64 mat_f64_mul(mat_f64 a, mat_f64 b);

mat_u8 mat_u8_had(mat_u8 a, mat_u8 b);
mat_u16 mat_u16_had(mat_u16 a, mat_u16 b);
mat_u32 mat_u32_had(mat_u32 a, mat_u32 b);
mat_u64 mat_u64_had(mat_u64 a, mat_u64 b);
mat_i8 mat_i8_had(mat_i8 a, mat_i8 b);
mat_i16 mat_i16_had(mat_i16 a, mat_i16 b);
mat_i32 mat_i32_had(mat_i32 a, mat_i32 b);
mat_i64 mat_i64_had(mat_i64 a, mat_i64 b);
mat_f32 mat_f32_had(mat_f32 a, mat_f32 b);
mat_f64 mat_f64_had(mat_f64 a, mat_f64 b);


inline u8 mat_u8_at(mat_u8 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline u16 mat_u16_at(mat_u16 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline u32 mat_u32_at(mat_u32 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline u64 mat_u64_at(mat_u64 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline i8 mat_i8_at(mat_i8 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline i16 mat_i16_at(mat_i16 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline i32 mat_i32_at(mat_i32 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline i64 mat_i64_at(mat_i64 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline f32 mat_f32_at(mat_f32 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline f64 mat_f64_at(mat_f64 m, usize row, usize col)
{
	return m.data[m.cols*row + col];
}

inline void mat_u8_set_val(mat_u8* m, usize row, usize col, u8 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_u16_set_val(mat_u16* m, usize row, usize col, u16 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_u32_set_val(mat_u32* m, usize row, usize col, u32 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_u64_set_val(mat_u64* m, usize row, usize col, u64 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_i8_set_val(mat_i8* m, usize row, usize col, i8 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_i16_set_val(mat_i16* m, usize row, usize col, i16 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_i32_set_val(mat_i32* m, usize row, usize col, i32 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_i64_set_val(mat_i64* m, usize row, usize col, i64 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_f32_set_val(mat_f32* m, usize row, usize col, f32 val)
{
	m->data[m->cols*row + col] = val;
}

inline void mat_f64_set_val(mat_f64* m, usize row, usize col, f64 val)
{
	m->data[m->cols*row + col] = val;
}
