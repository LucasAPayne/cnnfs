#pragma once


#ifdef __cplusplus
extern "C" {
#endif

typedef struct mat_f32 mat_f32;
typedef enum Device Device;

mat_f32 mat_f32_to(mat_f32 a, Device device);

mat_f32 mat_f32_zeros_gpu(usize rows, usize cols);

void mat_f32_add_gpu(mat_f32 a, mat_f32 b, mat_f32 c);

#ifdef __cplusplus
}
#endif
