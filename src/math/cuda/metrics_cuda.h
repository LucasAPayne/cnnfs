#pragma once

#include "device.h"
#include "types.h"
#include "vector.h"

f32 accuracy_score_gpu(vec<u32> y_true, vec<u32> y_pred);
