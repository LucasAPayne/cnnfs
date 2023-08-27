#pragma once

#include "math/cnnfs_math.h"
#include "util/types.h"

// Create spiral dataset by specifying the number of samples and classes
// Returns matrix of data and matrix of labels
void create_spiral_data(usize samples, u8 classes, mat_f32* out_data, mat_u8* out_labels);
