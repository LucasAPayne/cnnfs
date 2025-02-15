#pragma once

#include "activation.h"
#include "math/cnnfs_math.h"

// Dense, or fully-connected, layer, where each neuron in the layer
// is connected to each neuron from the previous and next layers
struct DenseLayer
{
    mat<f32> weights;
    vec<f32> biases;
    mat<f32> output;

    ActivationType activation;

    Device device;
};

DenseLayer dense_layer_init(size inputs, size neurons, ActivationType activation=Activation_Linear,
                            Device device=Device_CPU);

void dense_layer_forward(DenseLayer* layer, mat<f32> input);
