#pragma once

#include "math/cnnfs_math.h"

// Dense, or fully-connected, layer, where each neuron in the layer
// is connected to each neuron from the previous and next layers
struct DenseLayer
{
    mat<f32> weights;
    mat<f32> biases;
    mat<f32> output;
    Device device;
};

DenseLayer dense_layer_init(usize inputs, usize neurons, Device device=DEVICE_CPU);

void dense_layer_forward(DenseLayer* layer, mat<f32> input);
