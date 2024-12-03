#include "metrics.h"
#include "math/cuda/vector_cuda.h"

f32 accuracy_score(vec<u32> y_true, vec<u32> y_pred)
{
    ASSERT(y_true.elements == y_pred.elements);

    b32 y_true_was_on_gpu = (y_true.device == DEVICE_GPU);
    b32 y_pred_was_on_gpu = (y_pred.device == DEVICE_GPU);
    if (y_true_was_on_gpu)
        vec_to(&y_true, DEVICE_CPU);
    if (y_pred_was_on_gpu)
        vec_to(&y_true, DEVICE_CPU);

    u32 sum = 0;
    for (usize i = 0; i < y_true.elements; ++i)
        sum += (u32)(y_true.data[i] == y_pred.data[i]);
    
    f32 acc = (f32)sum/(f32)y_true.elements;
    return acc;
}
