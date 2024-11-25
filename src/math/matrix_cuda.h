#pragma once

typedef struct mat_f32 mat_f32;

#ifdef __cplusplus
extern "C" {
#endif

void mat_f32_add_gpu(mat_f32 a, mat_f32 b, mat_f32* c);

#ifdef __cplusplus
}
#endif
