#include "metrics.h"
#include "math/cuda/vector_cuda.h"

f32 accuracy_score(vec<u32> y_true, vec<u32> y_pred)
{
    ASSERT(y_true.elements == y_pred.elements, "Mismatch in number of true and predicted elements.\n");

    b32 y_true_was_on_gpu = (y_true.device == Device_GPU);
    b32 y_pred_was_on_gpu = (y_pred.device == Device_GPU);
    if (y_true_was_on_gpu)
        vec_to(&y_true, Device_CPU);
    if (y_pred_was_on_gpu)
        vec_to(&y_true, Device_CPU);

    u32 sum = 0;
    for (size i = 0; i < y_true.elements; ++i)
        sum += (u32)(y_true[i] == y_pred[i]);
    
    f32 acc = (f32)sum/(f32)y_true.elements;
    return acc;
}
