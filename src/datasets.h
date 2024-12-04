#pragma once

#include "math/cnnfs_math.h"
#include "util/types.h"

// Create spiral dataset by specifying the number of samples and classes
// Returns matrix of data and vector of labels
void create_spiral_data(usize samples, u32 classes, mat<f32>* out_data, vec<u32>* out_labels, Device device=Device_CPU);
