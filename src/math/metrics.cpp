#include "metrics.h"
#include "math/cuda/metrics_cuda.h"
#include "math/cuda/vector_cuda.h"

f32 accuracy_score(vec<u32> y_true, vec<u32> y_pred)
{
    ASSERT(y_true.device == y_pred.device,
           "True and predicted labels must be on the same device (y_true device: %hhu, y_pred device: %hhu).",
           y_true.device, y_pred.device);
    ASSERT(y_true.elements == y_pred.elements,
           "Mismatch in number of true and predicted elements (True: %llu, Pred: %llu).\n", y_true.elements, y_pred.elements);

    f32 acc = 0.0f;
    switch(y_true.device)
    {
        case Device_CPU:
        {
            u32 sum = 0;
            for (size i = 0; i < y_true.elements; ++i)
               sum += (u32)(y_true[i] == y_pred[i]);
            acc = (f32)sum/(f32)y_true.elements;
        } break;

        case Device_GPU: acc = accuracy_score_gpu(y_true, y_pred); break;

        default: log_invalid_device(y_true.device); break;
    }

    return acc;
}
