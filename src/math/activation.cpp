#include "activation.h"
#include "math/cuda/activation_cuda.h"
#include "cnnfs_math.h"

internal void relu_forward(mat<f32> inputs)
{
    switch (inputs.device)
    {
        case Device_CPU:
        {
            for (usize row = 0; row < inputs.rows; ++row)
            {
                for (usize col = 0; col < inputs.cols; ++col)
                    mat_set_val(inputs, row, col, max(0.0f, mat_at(inputs, row, col)));
            }
        } break;

        case Device_GPU: relu_forward_gpu(inputs);

        default: break;
    }
}

internal void relu_forward(mat<f64> inputs)
{
    switch (inputs.device)
    {
        case Device_CPU:
        {
            for (usize row = 0; row < inputs.rows; ++row)
            {
                for (usize col = 0; col < inputs.cols; ++col)
                    mat_set_val(inputs, row, col, max(0.0, mat_at(inputs, row, col)));
            }
        } break;

        case Device_GPU: relu_forward_gpu(inputs);

        default: break;
    }
}

internal void softmax_forward(mat<f32> inputs)
{
    exp_mat(inputs);
    vec<f32> sum = mat_sum(inputs);
    mat_scale(inputs, vec_reciprocal(sum));
}

internal void softmax_forward(mat<f64> inputs)
{
    exp_mat(inputs);
    vec<f64> sum = mat_sum(inputs);
    mat_scale(inputs, vec_reciprocal(sum));
}

void activation_forward(mat<f32> inputs, ActivationType activation)
{
    switch (activation)
    {
        case Activation_Linear: break;
        case Activation_ReLU: relu_forward(inputs); break;
        case Activation_Softmax: softmax_forward(inputs); break;
        default: break;
    }
}

void activation_forward(mat<f64> inputs, ActivationType activation)
{
    switch (activation)
    {
        case Activation_Linear: break;
        case Activation_ReLU: relu_forward(inputs); break;
        case Activation_Softmax: softmax_forward(inputs); break;
        default: break;
    }
}
