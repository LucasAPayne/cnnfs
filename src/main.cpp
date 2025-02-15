#include "cnnfs.h"

#include <stdio.h>

// TODO(lucas): Make sure every appropriate math function returns a value.

// TODO(lucas): Use more optimized reductions.
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf slide 35
// TODO(lucas): Is it possible to make a general reduction for matrices that works across rows or columns?
// TODO(lucas): Use shared memory in CUDA code where appropraite (e.g., matrix ops).

// TODO(lucas): Switch to growable arenas and get rid of individual vec/matrix allocations.
// TODO(lucas): Use scratch space for each pass over the neural network,
// or pre-allocate the memory for each output and have an option for multiplication to take in allocated memory.
// TODO(lucas): In softmax activation, sum vector can be popped (freed) after scaling input

int main(void)
{
    profiler_begin();
    rand_seed(123);

    Device device = Device_GPU;

    time_block_begin("Dataset Creation");
    mat<f32> data;
    vec<u32> labels;
    create_spiral_data(100, 3, &data, &labels, device);
    time_block_end();

    time_block_begin("Initialization");
    // 2 input features (x and y coordinates) and 3 output values
    DenseLayer dense1 = dense_layer_init(2, 3, Activation_ReLU, device);
    // Another dense layer
    DenseLayer dense2 = dense_layer_init(3, 3, Activation_ReLU, device);
    // Perform classification using softmax activation to generate a normalized confidence distribution.
    DenseLayer out = dense_layer_init(3, 3, Activation_Softmax, device);
    time_block_end();

    time_block_begin("Forward");
    dense_layer_forward(&dense1, data);
    dense_layer_forward(&dense2, dense1.output);
    dense_layer_forward(&out, dense2.output);
    time_block_end();

    mat_to(&out.output, Device_CPU);

    // TODO(lucas): Add a function to find the max value(s) of a matrix as a whole or along an axis,
    // and the index of the max value.
    time_block_begin("Prediction");
    vec<u32> pred = vec_zeros<u32>(labels.elements, Device_CPU);
    for (size i = 0; i < pred.elements; ++i)
    {
        vec<f32> confidence = mat_get_row(out.output, i);
        f32 max_confidence = -1.0f;
        u32 idx = 0;
        for (u32 j = 0; j < confidence.elements; ++j)
        {
            max_confidence = max(max_confidence, confidence[j]);
            if (max_confidence > confidence[j])
                idx = j;
        }
        pred[i] = idx;
    }
    time_block_end();

    vec_to(&labels, Device_CPU);
    time_block_begin("Accuracy");
    f32 acc = accuracy_score(labels, pred);
    time_block_end();

    profiler_end();

    if (out.output.rows <= 300)
        mat_print(out.output);
    printf("\nAccuracy: %.2f%%\n\n", acc*100.0f);
    profiler_print();

    getchar();
    return 0;
}

profiler_static_assert();
