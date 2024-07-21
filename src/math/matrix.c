#include "matrix.h"

mat_u8 mat_u8_init(usize rows, usize cols, u8* data)
{
	mat_u8 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_u16 mat_u16_init(usize rows, usize cols, u16* data)
{
	mat_u16 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_u32 mat_u32_init(usize rows, usize cols, u32* data)
{
	mat_u32 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_u64 mat_u64_init(usize rows, usize cols, u64* data)
{
	mat_u64 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_i8 mat_i8_init(usize rows, usize cols, i8* data)
{
	mat_i8 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_i16 mat_i16_init(usize rows, usize cols, i16* data)
{
	mat_i16 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_i32 mat_i32_init(usize rows, usize cols, i32* data)
{
	mat_i32 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_i64 mat_i64_init(usize rows, usize cols, i64* data)
{
	mat_i64 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_f32 mat_f32_init(usize rows, usize cols, f32* data)
{
	mat_f32 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_f64 mat_f64_init(usize rows, usize cols, f64* data)
{
	mat_f64 result = {0};
	result.rows = rows;
	result.cols = cols;
	result.data = data;
	
	return result;
}

mat_u8 mat_u8_zeros(usize rows, usize cols)
{
	mat_u8 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(u8));
	ASSERT(result.data);
	
	return result;
}

mat_u16 mat_u16_zeros(usize rows, usize cols)
{
	mat_u16 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(u16));
	ASSERT(result.data);
	
	return result;
}

mat_u32 mat_u32_zeros(usize rows, usize cols)
{
	mat_u32 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(u32));
	ASSERT(result.data);
	
	return result;
}

mat_u64 mat_u64_zeros(usize rows, usize cols)
{
	mat_u64 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(u64));
	ASSERT(result.data);
	
	return result;
}

mat_i8 mat_i8_zeros(usize rows, usize cols)
{
	mat_i8 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(i8));
	ASSERT(result.data);
	
	return result;
}

mat_i16 mat_i16_zeros(usize rows, usize cols)
{
	mat_i16 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(i16));
	ASSERT(result.data);
	
	return result;
}

mat_i32 mat_i32_zeros(usize rows, usize cols)
{
	mat_i32 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(i32));
	ASSERT(result.data);
	
	return result;
}

mat_i64 mat_i64_zeros(usize rows, usize cols)
{
	mat_i64 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(i64));
	ASSERT(result.data);
	
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

mat_f64 mat_f64_zeros(usize rows, usize cols)
{
	mat_f64 result = {0};
	result.rows = rows;
	result.cols = cols;
	
	result.data = calloc(rows*cols, sizeof(f64));
	ASSERT(result.data);
	
	return result;
}

mat_u8 mat_u8_full(usize rows, usize cols, u8 fill_value)
{
	mat_u8 m = mat_u8_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_u8_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_u16 mat_u16_full(usize rows, usize cols, u16 fill_value)
{
	mat_u16 m = mat_u16_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_u16_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_u32 mat_u32_full(usize rows, usize cols, u32 fill_value)
{
	mat_u32 m = mat_u32_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_u32_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_u64 mat_u64_full(usize rows, usize cols, u64 fill_value)
{
	mat_u64 m = mat_u64_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_u64_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_i8 mat_i8_full(usize rows, usize cols, i8 fill_value)
{
	mat_i8 m = mat_i8_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_i8_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_i16 mat_i16_full(usize rows, usize cols, i16 fill_value)
{
	mat_i16 m = mat_i16_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_i16_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_i32 mat_i32_full(usize rows, usize cols, i32 fill_value)
{
	mat_i32 m = mat_i32_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_i32_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_i64 mat_i64_full(usize rows, usize cols, i64 fill_value)
{
	mat_i64 m = mat_i64_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_i64_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_f32 mat_f32_full(usize rows, usize cols, f32 fill_value)
{
	mat_f32 m = mat_f32_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_f32_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

mat_f64 mat_f64_full(usize rows, usize cols, f64 fill_value)
{
	mat_f64 m = mat_f64_zeros(rows, cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
			mat_f64_set_val(& m, row, col, fill_value);
	}
	
	return m;
}

void mat_u8_set_row(mat_u8* m, vec_u8 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_u8_set_val(m, row, col, v.data[col]);
}

void mat_u16_set_row(mat_u16* m, vec_u16 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_u16_set_val(m, row, col, v.data[col]);
}

void mat_u32_set_row(mat_u32* m, vec_u32 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_u32_set_val(m, row, col, v.data[col]);
}

void mat_u64_set_row(mat_u64* m, vec_u64 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_u64_set_val(m, row, col, v.data[col]);
}

void mat_i8_set_row(mat_i8* m, vec_i8 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_i8_set_val(m, row, col, v.data[col]);
}

void mat_i16_set_row(mat_i16* m, vec_i16 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_i16_set_val(m, row, col, v.data[col]);
}

void mat_i32_set_row(mat_i32* m, vec_i32 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_i32_set_val(m, row, col, v.data[col]);
}

void mat_i64_set_row(mat_i64* m, vec_i64 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_i64_set_val(m, row, col, v.data[col]);
}

void mat_f32_set_row(mat_f32* m, vec_f32 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_f32_set_val(m, row, col, v.data[col]);
}

void mat_f64_set_row(mat_f64* m, vec_f64 v, usize row)
{
	
	
	ASSERT(m->cols == v.elements);
	
	for(usize col = 0; col < m->cols; ++col)
		mat_f64_set_val(m, row, col, v.data[col]);
}

void mat_u8_set_col(mat_u8* m, vec_u8 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_u8_set_val(m, row, col, v.data[row]);
}

void mat_u16_set_col(mat_u16* m, vec_u16 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_u16_set_val(m, row, col, v.data[row]);
}

void mat_u32_set_col(mat_u32* m, vec_u32 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_u32_set_val(m, row, col, v.data[row]);
}

void mat_u64_set_col(mat_u64* m, vec_u64 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_u64_set_val(m, row, col, v.data[row]);
}

void mat_i8_set_col(mat_i8* m, vec_i8 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_i8_set_val(m, row, col, v.data[row]);
}

void mat_i16_set_col(mat_i16* m, vec_i16 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_i16_set_val(m, row, col, v.data[row]);
}

void mat_i32_set_col(mat_i32* m, vec_i32 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_i32_set_val(m, row, col, v.data[row]);
}

void mat_i64_set_col(mat_i64* m, vec_i64 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_i64_set_val(m, row, col, v.data[row]);
}

void mat_f32_set_col(mat_f32* m, vec_f32 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_f32_set_val(m, row, col, v.data[row]);
}

void mat_f64_set_col(mat_f64* m, vec_f64 v, usize col)
{
	
	
	ASSERT(m->rows == v.elements);
	
	for(usize row = 0; row < m->rows; ++row)
		mat_f64_set_val(m, row, col, v.data[row]);
}

void mat_u8_set_row_range(mat_u8* m, vec_u8 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_u8_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_u16_set_row_range(mat_u16* m, vec_u16 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_u16_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_u32_set_row_range(mat_u32* m, vec_u32 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_u32_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_u64_set_row_range(mat_u64* m, vec_u64 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_u64_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_i8_set_row_range(mat_i8* m, vec_i8 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_i8_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_i16_set_row_range(mat_i16* m, vec_i16 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_i16_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_i32_set_row_range(mat_i32* m, vec_i32 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_i32_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_i64_set_row_range(mat_i64* m, vec_i64 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_i64_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_f32_set_row_range(mat_f32* m, vec_f32 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_f32_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_f64_set_row_range(mat_f64* m, vec_f64 v, usize row, usize row_offset)
{
	
	ASSERT(m->cols >= v.elements + row_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_f64_set_val(m, row, row_offset + i, v.data[i]);
}

void mat_u8_set_col_range(mat_u8* m, vec_u8 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_u8_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_u16_set_col_range(mat_u16* m, vec_u16 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_u16_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_u32_set_col_range(mat_u32* m, vec_u32 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_u32_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_u64_set_col_range(mat_u64* m, vec_u64 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_u64_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_i8_set_col_range(mat_i8* m, vec_i8 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_i8_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_i16_set_col_range(mat_i16* m, vec_i16 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_i16_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_i32_set_col_range(mat_i32* m, vec_i32 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_i32_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_i64_set_col_range(mat_i64* m, vec_i64 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_i64_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_f32_set_col_range(mat_f32* m, vec_f32 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_f32_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_f64_set_col_range(mat_f64* m, vec_f64 v, usize col, usize col_offset)
{
	
	ASSERT(m->rows >= v.elements + col_offset);
	
	for(usize i = 0; i < v.elements; ++i)
		mat_f64_set_val(m, col_offset + i, col, v.data[i]);
}

void mat_u8_print(mat_u8 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%hhu" , mat_u8_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*hhu" , width, mat_u8_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_u16_print(mat_u16 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%hu" , mat_u16_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*hu" , width, mat_u16_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_u32_print(mat_u32 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%u" , mat_u32_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*u" , width, mat_u32_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_u64_print(mat_u64 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%llu" , mat_u64_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*llu" , width, mat_u64_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_i8_print(mat_i8 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%hhd" , mat_i8_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*hhd" , width, mat_i8_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_i16_print(mat_i16 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%hd" , mat_i16_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*hd" , width, mat_i16_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_i32_print(mat_i32 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%d" , mat_i32_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*d" , width, mat_i32_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_i64_print(mat_i64 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%lld" , mat_i64_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*lld" , width, mat_i64_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_f32_print(mat_f32 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%f" , mat_f32_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*f" , width, mat_f32_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

void mat_f64_print(mat_f64 m)
{
	int width = 0;
	
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			int w = snprintf(NULL, 0 , "%f" , mat_f64_at(m, row, col));
			if(width < w)
				width = w;
		}
	}
	
	
	printf("[");
	for(usize row = 0; row < m.rows; ++row)
	{
		printf("[");
		for(usize col = 0; col < m.cols; ++col)
		{
			if(col != 0) printf(", ");
			printf("%*f" , width, mat_f64_at(m, row, col));
		}
		printf("]");
		
		if(row < m.rows - 1)
			printf(",\n");
	}
	printf("]");
}

mat_u8 mat_u8_scale(mat_u8 m, u8 value)
{
	mat_u8 result = mat_u8_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			u8 element = mat_u8_at(m, row, col);
			mat_u8_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_u16 mat_u16_scale(mat_u16 m, u16 value)
{
	mat_u16 result = mat_u16_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			u16 element = mat_u16_at(m, row, col);
			mat_u16_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_u32 mat_u32_scale(mat_u32 m, u32 value)
{
	mat_u32 result = mat_u32_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			u32 element = mat_u32_at(m, row, col);
			mat_u32_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_u64 mat_u64_scale(mat_u64 m, u64 value)
{
	mat_u64 result = mat_u64_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			u64 element = mat_u64_at(m, row, col);
			mat_u64_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_i8 mat_i8_scale(mat_i8 m, i8 value)
{
	mat_i8 result = mat_i8_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			i8 element = mat_i8_at(m, row, col);
			mat_i8_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_i16 mat_i16_scale(mat_i16 m, i16 value)
{
	mat_i16 result = mat_i16_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			i16 element = mat_i16_at(m, row, col);
			mat_i16_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_i32 mat_i32_scale(mat_i32 m, i32 value)
{
	mat_i32 result = mat_i32_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			i32 element = mat_i32_at(m, row, col);
			mat_i32_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_i64 mat_i64_scale(mat_i64 m, i64 value)
{
	mat_i64 result = mat_i64_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			i64 element = mat_i64_at(m, row, col);
			mat_i64_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_f32 mat_f32_scale(mat_f32 m, f32 value)
{
	mat_f32 result = mat_f32_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			f32 element = mat_f32_at(m, row, col);
			mat_f32_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

mat_f64 mat_f64_scale(mat_f64 m, f64 value)
{
	mat_f64 result = mat_f64_zeros(m.rows, m.cols);
	
	for(usize row = 0; row < m.rows; ++row)
	{
		for(usize col = 0; col < m.cols; ++col)
		{
			f64 element = mat_f64_at(m, row, col);
			mat_f64_set_val(& result, row, col, element*value);
		}
	}
	
	return result;
}

internal mat_u8 mat_u8_stretch_cols(mat_u8 orig, mat_u8 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_u8 result = mat_u8_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			u8 val = mat_u8_at(orig, i, 0);
			mat_u8_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_u16 mat_u16_stretch_cols(mat_u16 orig, mat_u16 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_u16 result = mat_u16_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			u16 val = mat_u16_at(orig, i, 0);
			mat_u16_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_u32 mat_u32_stretch_cols(mat_u32 orig, mat_u32 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_u32 result = mat_u32_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			u32 val = mat_u32_at(orig, i, 0);
			mat_u32_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_u64 mat_u64_stretch_cols(mat_u64 orig, mat_u64 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_u64 result = mat_u64_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			u64 val = mat_u64_at(orig, i, 0);
			mat_u64_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_i8 mat_i8_stretch_cols(mat_i8 orig, mat_i8 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_i8 result = mat_i8_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			i8 val = mat_i8_at(orig, i, 0);
			mat_i8_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_i16 mat_i16_stretch_cols(mat_i16 orig, mat_i16 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_i16 result = mat_i16_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			i16 val = mat_i16_at(orig, i, 0);
			mat_i16_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_i32 mat_i32_stretch_cols(mat_i32 orig, mat_i32 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_i32 result = mat_i32_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			i32 val = mat_i32_at(orig, i, 0);
			mat_i32_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_i64 mat_i64_stretch_cols(mat_i64 orig, mat_i64 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_i64 result = mat_i64_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			i64 val = mat_i64_at(orig, i, 0);
			mat_i64_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_f32 mat_f32_stretch_cols(mat_f32 orig, mat_f32 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_f32 result = mat_f32_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			f32 val = mat_f32_at(orig, i, 0);
			mat_f32_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_f64 mat_f64_stretch_cols(mat_f64 orig, mat_f64 target)
{
	
	ASSERT(orig.rows == target.rows);
	mat_f64 result = mat_f64_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			f64 val = mat_f64_at(orig, i, 0);
			mat_f64_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

internal mat_u8 mat_u8_stretch_rows(mat_u8 orig, mat_u8 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_u8 result = mat_u8_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			u8 val = mat_u8_at(orig, 0 , j);
			mat_u8_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_u16 mat_u16_stretch_rows(mat_u16 orig, mat_u16 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_u16 result = mat_u16_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			u16 val = mat_u16_at(orig, 0 , j);
			mat_u16_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_u32 mat_u32_stretch_rows(mat_u32 orig, mat_u32 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_u32 result = mat_u32_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			u32 val = mat_u32_at(orig, 0 , j);
			mat_u32_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_u64 mat_u64_stretch_rows(mat_u64 orig, mat_u64 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_u64 result = mat_u64_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			u64 val = mat_u64_at(orig, 0 , j);
			mat_u64_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_i8 mat_i8_stretch_rows(mat_i8 orig, mat_i8 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_i8 result = mat_i8_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			i8 val = mat_i8_at(orig, 0 , j);
			mat_i8_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_i16 mat_i16_stretch_rows(mat_i16 orig, mat_i16 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_i16 result = mat_i16_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			i16 val = mat_i16_at(orig, 0 , j);
			mat_i16_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_i32 mat_i32_stretch_rows(mat_i32 orig, mat_i32 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_i32 result = mat_i32_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			i32 val = mat_i32_at(orig, 0 , j);
			mat_i32_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_i64 mat_i64_stretch_rows(mat_i64 orig, mat_i64 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_i64 result = mat_i64_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			i64 val = mat_i64_at(orig, 0 , j);
			mat_i64_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_f32 mat_f32_stretch_rows(mat_f32 orig, mat_f32 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_f32 result = mat_f32_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			f32 val = mat_f32_at(orig, 0 , j);
			mat_f32_set_val(& result, i, j, val);
		}
	}
	return result;
}

internal mat_f64 mat_f64_stretch_rows(mat_f64 orig, mat_f64 target)
{
	
	ASSERT(orig.cols == target.cols);
	mat_f64 result = mat_f64_zeros(target.rows, target.cols);
	for(usize i = 0; i < target.rows; ++i)
	{
		for(usize j = 0; j < target.cols; ++j)
		{
			f64 val = mat_f64_at(orig, 0 , j);
			mat_f64_set_val(& result, i, j, val);
		}
	}
	return result;
}

mat_u8 mat_u8_add(mat_u8 a, mat_u8 b)
{
	
	b32 a_col_vec_u8 = a.cols == 1;
	b32 a_row_vec_u8 = a.rows == 1;
	b32 b_col_vec_u8 = b.cols == 1;
	b32 b_row_vec_u8 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_u8 ||  b_col_vec_u8) && (a.rows == b.rows))
	||  ((a_row_vec_u8 ||  b_row_vec_u8) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_u8)
			b = mat_u8_stretch_rows(b, a);
		else if(b_col_vec_u8)
			b = mat_u8_stretch_cols(b, a);
		else if(a_row_vec_u8)
			a = mat_u8_stretch_cols(a, b);
		else if(a_col_vec_u8)
			a = mat_u8_stretch_cols(a, b);
	}
	
	mat_u8 result = mat_u8_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			u8 val = mat_u8_at(a, i, j) + mat_u8_at(b, i, j);
			mat_u8_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_u16 mat_u16_add(mat_u16 a, mat_u16 b)
{
	
	b32 a_col_vec_u16 = a.cols == 1;
	b32 a_row_vec_u16 = a.rows == 1;
	b32 b_col_vec_u16 = b.cols == 1;
	b32 b_row_vec_u16 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_u16 ||  b_col_vec_u16) && (a.rows == b.rows))
	||  ((a_row_vec_u16 ||  b_row_vec_u16) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_u16)
			b = mat_u16_stretch_rows(b, a);
		else if(b_col_vec_u16)
			b = mat_u16_stretch_cols(b, a);
		else if(a_row_vec_u16)
			a = mat_u16_stretch_cols(a, b);
		else if(a_col_vec_u16)
			a = mat_u16_stretch_cols(a, b);
	}
	
	mat_u16 result = mat_u16_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			u16 val = mat_u16_at(a, i, j) + mat_u16_at(b, i, j);
			mat_u16_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_u32 mat_u32_add(mat_u32 a, mat_u32 b)
{
	
	b32 a_col_vec_u32 = a.cols == 1;
	b32 a_row_vec_u32 = a.rows == 1;
	b32 b_col_vec_u32 = b.cols == 1;
	b32 b_row_vec_u32 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_u32 ||  b_col_vec_u32) && (a.rows == b.rows))
	||  ((a_row_vec_u32 ||  b_row_vec_u32) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_u32)
			b = mat_u32_stretch_rows(b, a);
		else if(b_col_vec_u32)
			b = mat_u32_stretch_cols(b, a);
		else if(a_row_vec_u32)
			a = mat_u32_stretch_cols(a, b);
		else if(a_col_vec_u32)
			a = mat_u32_stretch_cols(a, b);
	}
	
	mat_u32 result = mat_u32_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			u32 val = mat_u32_at(a, i, j) + mat_u32_at(b, i, j);
			mat_u32_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_u64 mat_u64_add(mat_u64 a, mat_u64 b)
{
	
	b32 a_col_vec_u64 = a.cols == 1;
	b32 a_row_vec_u64 = a.rows == 1;
	b32 b_col_vec_u64 = b.cols == 1;
	b32 b_row_vec_u64 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_u64 ||  b_col_vec_u64) && (a.rows == b.rows))
	||  ((a_row_vec_u64 ||  b_row_vec_u64) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_u64)
			b = mat_u64_stretch_rows(b, a);
		else if(b_col_vec_u64)
			b = mat_u64_stretch_cols(b, a);
		else if(a_row_vec_u64)
			a = mat_u64_stretch_cols(a, b);
		else if(a_col_vec_u64)
			a = mat_u64_stretch_cols(a, b);
	}
	
	mat_u64 result = mat_u64_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			u64 val = mat_u64_at(a, i, j) + mat_u64_at(b, i, j);
			mat_u64_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_i8 mat_i8_add(mat_i8 a, mat_i8 b)
{
	
	b32 a_col_vec_i8 = a.cols == 1;
	b32 a_row_vec_i8 = a.rows == 1;
	b32 b_col_vec_i8 = b.cols == 1;
	b32 b_row_vec_i8 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_i8 ||  b_col_vec_i8) && (a.rows == b.rows))
	||  ((a_row_vec_i8 ||  b_row_vec_i8) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_i8)
			b = mat_i8_stretch_rows(b, a);
		else if(b_col_vec_i8)
			b = mat_i8_stretch_cols(b, a);
		else if(a_row_vec_i8)
			a = mat_i8_stretch_cols(a, b);
		else if(a_col_vec_i8)
			a = mat_i8_stretch_cols(a, b);
	}
	
	mat_i8 result = mat_i8_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			i8 val = mat_i8_at(a, i, j) + mat_i8_at(b, i, j);
			mat_i8_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_i16 mat_i16_add(mat_i16 a, mat_i16 b)
{
	
	b32 a_col_vec_i16 = a.cols == 1;
	b32 a_row_vec_i16 = a.rows == 1;
	b32 b_col_vec_i16 = b.cols == 1;
	b32 b_row_vec_i16 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_i16 ||  b_col_vec_i16) && (a.rows == b.rows))
	||  ((a_row_vec_i16 ||  b_row_vec_i16) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_i16)
			b = mat_i16_stretch_rows(b, a);
		else if(b_col_vec_i16)
			b = mat_i16_stretch_cols(b, a);
		else if(a_row_vec_i16)
			a = mat_i16_stretch_cols(a, b);
		else if(a_col_vec_i16)
			a = mat_i16_stretch_cols(a, b);
	}
	
	mat_i16 result = mat_i16_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			i16 val = mat_i16_at(a, i, j) + mat_i16_at(b, i, j);
			mat_i16_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_i32 mat_i32_add(mat_i32 a, mat_i32 b)
{
	
	b32 a_col_vec_i32 = a.cols == 1;
	b32 a_row_vec_i32 = a.rows == 1;
	b32 b_col_vec_i32 = b.cols == 1;
	b32 b_row_vec_i32 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_i32 ||  b_col_vec_i32) && (a.rows == b.rows))
	||  ((a_row_vec_i32 ||  b_row_vec_i32) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_i32)
			b = mat_i32_stretch_rows(b, a);
		else if(b_col_vec_i32)
			b = mat_i32_stretch_cols(b, a);
		else if(a_row_vec_i32)
			a = mat_i32_stretch_cols(a, b);
		else if(a_col_vec_i32)
			a = mat_i32_stretch_cols(a, b);
	}
	
	mat_i32 result = mat_i32_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			i32 val = mat_i32_at(a, i, j) + mat_i32_at(b, i, j);
			mat_i32_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_i64 mat_i64_add(mat_i64 a, mat_i64 b)
{
	
	b32 a_col_vec_i64 = a.cols == 1;
	b32 a_row_vec_i64 = a.rows == 1;
	b32 b_col_vec_i64 = b.cols == 1;
	b32 b_row_vec_i64 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_i64 ||  b_col_vec_i64) && (a.rows == b.rows))
	||  ((a_row_vec_i64 ||  b_row_vec_i64) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_i64)
			b = mat_i64_stretch_rows(b, a);
		else if(b_col_vec_i64)
			b = mat_i64_stretch_cols(b, a);
		else if(a_row_vec_i64)
			a = mat_i64_stretch_cols(a, b);
		else if(a_col_vec_i64)
			a = mat_i64_stretch_cols(a, b);
	}
	
	mat_i64 result = mat_i64_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			i64 val = mat_i64_at(a, i, j) + mat_i64_at(b, i, j);
			mat_i64_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_f32 mat_f32_add(mat_f32 a, mat_f32 b)
{
	
	b32 a_col_vec_f32 = a.cols == 1;
	b32 a_row_vec_f32 = a.rows == 1;
	b32 b_col_vec_f32 = b.cols == 1;
	b32 b_row_vec_f32 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_f32 ||  b_col_vec_f32) && (a.rows == b.rows))
	||  ((a_row_vec_f32 ||  b_row_vec_f32) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_f32)
			b = mat_f32_stretch_rows(b, a);
		else if(b_col_vec_f32)
			b = mat_f32_stretch_cols(b, a);
		else if(a_row_vec_f32)
			a = mat_f32_stretch_cols(a, b);
		else if(a_col_vec_f32)
			a = mat_f32_stretch_cols(a, b);
	}
	
	mat_f32 result = mat_f32_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			f32 val = mat_f32_at(a, i, j) + mat_f32_at(b, i, j);
			mat_f32_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_f64 mat_f64_add(mat_f64 a, mat_f64 b)
{
	
	b32 a_col_vec_f64 = a.cols == 1;
	b32 a_row_vec_f64 = a.rows == 1;
	b32 b_col_vec_f64 = b.cols == 1;
	b32 b_row_vec_f64 = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	||  ((a_col_vec_f64 ||  b_col_vec_f64) && (a.rows == b.rows))
	||  ((a_row_vec_f64 ||  b_row_vec_f64) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
	
	
	if((a.rows != b.rows) ||  (a.cols != b.cols))
	{
		if(b_row_vec_f64)
			b = mat_f64_stretch_rows(b, a);
		else if(b_col_vec_f64)
			b = mat_f64_stretch_cols(b, a);
		else if(a_row_vec_f64)
			a = mat_f64_stretch_cols(a, b);
		else if(a_col_vec_f64)
			a = mat_f64_stretch_cols(a, b);
	}
	
	mat_f64 result = mat_f64_zeros(a.rows, a.cols);
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			f64 val = mat_f64_at(a, i, j) + mat_f64_at(b, i, j);
			mat_f64_set_val(& result, i, j, val);
		}
	}
	
	return result;
}

mat_u8 mat_u8_mul(mat_u8 a, mat_u8 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_u8 result = mat_u8_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			u8 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_u8_at(a, i, k) * mat_u8_at(b, k, j);
			
			mat_u8_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_u16 mat_u16_mul(mat_u16 a, mat_u16 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_u16 result = mat_u16_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			u16 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_u16_at(a, i, k) * mat_u16_at(b, k, j);
			
			mat_u16_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_u32 mat_u32_mul(mat_u32 a, mat_u32 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_u32 result = mat_u32_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			u32 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_u32_at(a, i, k) * mat_u32_at(b, k, j);
			
			mat_u32_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_u64 mat_u64_mul(mat_u64 a, mat_u64 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_u64 result = mat_u64_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			u64 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_u64_at(a, i, k) * mat_u64_at(b, k, j);
			
			mat_u64_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_i8 mat_i8_mul(mat_i8 a, mat_i8 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_i8 result = mat_i8_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			i8 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_i8_at(a, i, k) * mat_i8_at(b, k, j);
			
			mat_i8_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_i16 mat_i16_mul(mat_i16 a, mat_i16 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_i16 result = mat_i16_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			i16 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_i16_at(a, i, k) * mat_i16_at(b, k, j);
			
			mat_i16_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_i32 mat_i32_mul(mat_i32 a, mat_i32 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_i32 result = mat_i32_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			i32 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_i32_at(a, i, k) * mat_i32_at(b, k, j);
			
			mat_i32_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_i64 mat_i64_mul(mat_i64 a, mat_i64 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_i64 result = mat_i64_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			i64 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_i64_at(a, i, k) * mat_i64_at(b, k, j);
			
			mat_i64_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_f32 mat_f32_mul(mat_f32 a, mat_f32 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_f32 result = mat_f32_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			f32 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_f32_at(a, i, k) * mat_f32_at(b, k, j);
			
			mat_f32_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_f64 mat_f64_mul(mat_f64 a, mat_f64 b)
{
	
	ASSERT(a.cols == b.rows);
	
	
	mat_f64 result = mat_f64_zeros(a.rows, b.cols);
	
	
	for(usize i = 0; i < result.rows; ++i)
	{
		for(usize j = 0; j < result.cols; ++j)
		{
			f64 sum = 0;
			for(usize k = 0; k < a.cols; ++k)
				sum += mat_f64_at(a, i, k) * mat_f64_at(b, k, j);
			
			mat_f64_set_val(& result, i, j, sum);
		}
	}
	return result;
}

mat_u8 mat_u8_had(mat_u8 a, mat_u8 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_u8 result = mat_u8_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			u8 val = mat_u8_at(a, row, col) * mat_u8_at(b, row, col);
			mat_u8_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_u16 mat_u16_had(mat_u16 a, mat_u16 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_u16 result = mat_u16_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			u16 val = mat_u16_at(a, row, col) * mat_u16_at(b, row, col);
			mat_u16_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_u32 mat_u32_had(mat_u32 a, mat_u32 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_u32 result = mat_u32_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			u32 val = mat_u32_at(a, row, col) * mat_u32_at(b, row, col);
			mat_u32_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_u64 mat_u64_had(mat_u64 a, mat_u64 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_u64 result = mat_u64_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			u64 val = mat_u64_at(a, row, col) * mat_u64_at(b, row, col);
			mat_u64_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_i8 mat_i8_had(mat_i8 a, mat_i8 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_i8 result = mat_i8_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			i8 val = mat_i8_at(a, row, col) * mat_i8_at(b, row, col);
			mat_i8_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_i16 mat_i16_had(mat_i16 a, mat_i16 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_i16 result = mat_i16_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			i16 val = mat_i16_at(a, row, col) * mat_i16_at(b, row, col);
			mat_i16_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_i32 mat_i32_had(mat_i32 a, mat_i32 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_i32 result = mat_i32_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			i32 val = mat_i32_at(a, row, col) * mat_i32_at(b, row, col);
			mat_i32_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_i64 mat_i64_had(mat_i64 a, mat_i64 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_i64 result = mat_i64_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			i64 val = mat_i64_at(a, row, col) * mat_i64_at(b, row, col);
			mat_i64_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_f32 mat_f32_had(mat_f32 a, mat_f32 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_f32 result = mat_f32_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			f32 val = mat_f32_at(a, row, col) * mat_f32_at(b, row, col);
			mat_f32_set_val(& result, row, col, val);
		}
	}
	
	return result;
}

mat_f64 mat_f64_had(mat_f64 a, mat_f64 b)
{
	ASSERT(a.rows == b.rows);
	ASSERT(a.cols == b.cols);
	
	mat_f64 result = mat_f64_zeros(a.rows, a.cols);
	
	for(usize row = 0; row < result.rows; ++row)
	{
		for(usize col = 0; col < result.cols; ++col)
		{
			f64 val = mat_f64_at(a, row, col) * mat_f64_at(b, row, col);
			mat_f64_set_val(& result, row, col, val);
		}
	}
	
	return result;
}
