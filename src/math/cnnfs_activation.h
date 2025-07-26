#pragma once

#include "matrix.h"

enum ActivationType : u8
{
    Activation_Linear = 0,
    Activation_ReLU,
    Activation_Softmax
};

/* Rectified Linear Unit (ReLU) activation function.
 * For each input, returns max(0, input).
 */
internal void relu_forward(mat<f32> inputs);
internal void relu_forward(mat<f64> inputs);

internal void softmax_forward(mat<f32> inputs);
internal void softmax_forward(mat<f64> inputs);

// Call any activation function by specifying an ActivationType.
void activation_forward(mat<f32> inputs, ActivationType activation);
void activation_forward(mat<f64> inputs, ActivationType activation);
