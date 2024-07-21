#include "vector.h"

vec_u8 vec_u8_init(usize elements, u8* data)
{
	vec_u8 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_u16 vec_u16_init(usize elements, u16* data)
{
	vec_u16 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_u32 vec_u32_init(usize elements, u32* data)
{
	vec_u32 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_u64 vec_u64_init(usize elements, u64* data)
{
	vec_u64 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_i8 vec_i8_init(usize elements, i8* data)
{
	vec_i8 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_i16 vec_i16_init(usize elements, i16* data)
{
	vec_i16 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_i32 vec_i32_init(usize elements, i32* data)
{
	vec_i32 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_i64 vec_i64_init(usize elements, i64* data)
{
	vec_i64 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_f32 vec_f32_init(usize elements, f32* data)
{
	vec_f32 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_f64 vec_f64_init(usize elements, f64* data)
{
	vec_f64 result = {0};
	result.elements = elements;
	result.data = data;
	
	return result;
}

vec_u8 vec_u8_zeros(usize elements)
{
	vec_u8 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(u8));
	ASSERT(result.data);
	
	return result;
}

vec_u16 vec_u16_zeros(usize elements)
{
	vec_u16 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(u16));
	ASSERT(result.data);
	
	return result;
}

vec_u32 vec_u32_zeros(usize elements)
{
	vec_u32 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(u32));
	ASSERT(result.data);
	
	return result;
}

vec_u64 vec_u64_zeros(usize elements)
{
	vec_u64 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(u64));
	ASSERT(result.data);
	
	return result;
}

vec_i8 vec_i8_zeros(usize elements)
{
	vec_i8 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(i8));
	ASSERT(result.data);
	
	return result;
}

vec_i16 vec_i16_zeros(usize elements)
{
	vec_i16 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(i16));
	ASSERT(result.data);
	
	return result;
}

vec_i32 vec_i32_zeros(usize elements)
{
	vec_i32 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(i32));
	ASSERT(result.data);
	
	return result;
}

vec_i64 vec_i64_zeros(usize elements)
{
	vec_i64 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(i64));
	ASSERT(result.data);
	
	return result;
}

vec_f32 vec_f32_zeros(usize elements)
{
	vec_f32 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(f32));
	ASSERT(result.data);
	
	return result;
}

vec_f64 vec_f64_zeros(usize elements)
{
	vec_f64 result = {0};
	result.elements = elements;
	
	result.data = calloc(elements, sizeof(f64));
	ASSERT(result.data);
	
	return result;
}

vec_u8 vec_u8_full(usize elements, u8 fill_value)
{
	vec_u8 v = vec_u8_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_u16 vec_u16_full(usize elements, u16 fill_value)
{
	vec_u16 v = vec_u16_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_u32 vec_u32_full(usize elements, u32 fill_value)
{
	vec_u32 v = vec_u32_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_u64 vec_u64_full(usize elements, u64 fill_value)
{
	vec_u64 v = vec_u64_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_i8 vec_i8_full(usize elements, i8 fill_value)
{
	vec_i8 v = vec_i8_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_i16 vec_i16_full(usize elements, i16 fill_value)
{
	vec_i16 v = vec_i16_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_i32 vec_i32_full(usize elements, i32 fill_value)
{
	vec_i32 v = vec_i32_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_i64 vec_i64_full(usize elements, i64 fill_value)
{
	vec_i64 v = vec_i64_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_f32 vec_f32_full(usize elements, f32 fill_value)
{
	vec_f32 v = vec_f32_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

vec_f64 vec_f64_full(usize elements, f64 fill_value)
{
	vec_f64 v = vec_f64_zeros(elements);
	
	for(usize i = 0; i < v.elements; ++i)
		v.data[i] = fill_value;
	
	return v;
}

void vec_u8_set_range(vec_u8* v, vec_u8 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_u16_set_range(vec_u16* v, vec_u16 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_u32_set_range(vec_u32* v, vec_u32 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_u64_set_range(vec_u64* v, vec_u64 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_i8_set_range(vec_i8* v, vec_i8 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_i16_set_range(vec_i16* v, vec_i16 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_i32_set_range(vec_i32* v, vec_i32 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_i64_set_range(vec_i64* v, vec_i64 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_f32_set_range(vec_f32* v, vec_f32 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_f64_set_range(vec_f64* v, vec_f64 data, usize begin)
{
	
	ASSERT(v->elements >= data.elements + begin);
	
	for(usize i = 0; i < data.elements; ++i)
		v->data[begin + i] = data.data[i];
}

void vec_u8_print(vec_u8 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%hhu" , v.data[i]);
	}
	printf("]\n");
}

void vec_u16_print(vec_u16 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%hu" , v.data[i]);
	}
	printf("]\n");
}

void vec_u32_print(vec_u32 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%u" , v.data[i]);
	}
	printf("]\n");
}

void vec_u64_print(vec_u64 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%llu" , v.data[i]);
	}
	printf("]\n");
}

void vec_i8_print(vec_i8 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%hhd" , v.data[i]);
	}
	printf("]\n");
}

void vec_i16_print(vec_i16 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%hd" , v.data[i]);
	}
	printf("]\n");
}

void vec_i32_print(vec_i32 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%d" , v.data[i]);
	}
	printf("]\n");
}

void vec_i64_print(vec_i64 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%lld" , v.data[i]);
	}
	printf("]\n");
}

void vec_f32_print(vec_f32 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%f" , v.data[i]);
	}
	printf("]\n");
}

void vec_f64_print(vec_f64 v)
{
	printf("[");
	for(usize i = 0; i < v.elements; ++i)
	{
		if(i != 0) printf(", ");
		printf("%f" , v.data[i]);
	}
	printf("]\n");
}

vec_u8 vec_u8_add(vec_u8 a, vec_u8 b)
{
	ASSERT(a.elements == b.elements);
	vec_u8 result = vec_u8_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_u16 vec_u16_add(vec_u16 a, vec_u16 b)
{
	ASSERT(a.elements == b.elements);
	vec_u16 result = vec_u16_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_u32 vec_u32_add(vec_u32 a, vec_u32 b)
{
	ASSERT(a.elements == b.elements);
	vec_u32 result = vec_u32_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_u64 vec_u64_add(vec_u64 a, vec_u64 b)
{
	ASSERT(a.elements == b.elements);
	vec_u64 result = vec_u64_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_i8 vec_i8_add(vec_i8 a, vec_i8 b)
{
	ASSERT(a.elements == b.elements);
	vec_i8 result = vec_i8_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_i16 vec_i16_add(vec_i16 a, vec_i16 b)
{
	ASSERT(a.elements == b.elements);
	vec_i16 result = vec_i16_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_i32 vec_i32_add(vec_i32 a, vec_i32 b)
{
	ASSERT(a.elements == b.elements);
	vec_i32 result = vec_i32_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_i64 vec_i64_add(vec_i64 a, vec_i64 b)
{
	ASSERT(a.elements == b.elements);
	vec_i64 result = vec_i64_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_f32 vec_f32_add(vec_f32 a, vec_f32 b)
{
	ASSERT(a.elements == b.elements);
	vec_f32 result = vec_f32_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_f64 vec_f64_add(vec_f64 a, vec_f64 b)
{
	ASSERT(a.elements == b.elements);
	vec_f64 result = vec_f64_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] + b.data[i];
	
	return result;
}

vec_u8 vec_u8_scale(vec_u8 v, u8 a)
{
	vec_u8 result = vec_u8_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_u16 vec_u16_scale(vec_u16 v, u16 a)
{
	vec_u16 result = vec_u16_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_u32 vec_u32_scale(vec_u32 v, u32 a)
{
	vec_u32 result = vec_u32_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_u64 vec_u64_scale(vec_u64 v, u64 a)
{
	vec_u64 result = vec_u64_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_i8 vec_i8_scale(vec_i8 v, i8 a)
{
	vec_i8 result = vec_i8_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_i16 vec_i16_scale(vec_i16 v, i16 a)
{
	vec_i16 result = vec_i16_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_i32 vec_i32_scale(vec_i32 v, i32 a)
{
	vec_i32 result = vec_i32_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_i64 vec_i64_scale(vec_i64 v, i64 a)
{
	vec_i64 result = vec_i64_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_f32 vec_f32_scale(vec_f32 v, f32 a)
{
	vec_f32 result = vec_f32_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_f64 vec_f64_scale(vec_f64 v, f64 a)
{
	vec_f64 result = vec_f64_zeros(v.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a*v.data[i];
	
	return result;
}

vec_u8 vec_u8_had(vec_u8 a, vec_u8 b)
{
	ASSERT(a.elements == b.elements);
	vec_u8 result = vec_u8_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_u16 vec_u16_had(vec_u16 a, vec_u16 b)
{
	ASSERT(a.elements == b.elements);
	vec_u16 result = vec_u16_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_u32 vec_u32_had(vec_u32 a, vec_u32 b)
{
	ASSERT(a.elements == b.elements);
	vec_u32 result = vec_u32_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_u64 vec_u64_had(vec_u64 a, vec_u64 b)
{
	ASSERT(a.elements == b.elements);
	vec_u64 result = vec_u64_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_i8 vec_i8_had(vec_i8 a, vec_i8 b)
{
	ASSERT(a.elements == b.elements);
	vec_i8 result = vec_i8_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_i16 vec_i16_had(vec_i16 a, vec_i16 b)
{
	ASSERT(a.elements == b.elements);
	vec_i16 result = vec_i16_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_i32 vec_i32_had(vec_i32 a, vec_i32 b)
{
	ASSERT(a.elements == b.elements);
	vec_i32 result = vec_i32_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_i64 vec_i64_had(vec_i64 a, vec_i64 b)
{
	ASSERT(a.elements == b.elements);
	vec_i64 result = vec_i64_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_f32 vec_f32_had(vec_f32 a, vec_f32 b)
{
	ASSERT(a.elements == b.elements);
	vec_f32 result = vec_f32_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}

vec_f64 vec_f64_had(vec_f64 a, vec_f64 b)
{
	ASSERT(a.elements == b.elements);
	vec_f64 result = vec_f64_zeros(a.elements);
	
	for(usize i = 0; i < result.elements; ++i)
		result.data[i] = a.data[i] * b.data[i];
	
	return result;
}
