#pragma once

#include "math/cnnfs_math.h"

// Dense, or fully-connected, layer, where each neuron in the layer
// is connected to each neuron from the previous and next layers
typedef struct dense_layer
{
    mat_f32 weights;
    mat_f32 biases;
    mat_f32 output;
} dense_layer;

// dense_layer dense_layer_init(usize inputs, usize neurons);
void dense_layer_forward(dense_layer* layer, mat_f32 input);
