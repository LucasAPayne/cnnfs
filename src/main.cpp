#include "cnnfs.h"

#include <stdio.h>

// TODO(lucas): Add operator overloading.
// TODO(lucas): Drop mat/vec prefixes for functions and just use overloading and namespace nn?
// TODO(lucas): Matrix and vector functions should still operate in-place, but also return the value for more expressiveness.
// TODO(lucas): Try to eliminate copying data between host and device in mat_sum_gpu and mat_scale_gpu
// TODO(lucas): Look into CUDA warnings
// TODO(lucas): Look into another way of specifying that some function templates are only valid for certain types,
// besides just making multiple copies, to make for easier maintenance.
// TODO(lucas): Some math and other functions should have an option of operating in-place or returning a copy.
// TODO(lucas): Should the math functions just be rolled into vector.h?
// TODO(lucas): Add RNG for GPU.
// TODO(lucas): Add profiling.
// TODO(lucas): Add logging.
// TODO(lucas): Log invalid device errors and other assertions.
// TODO(lucas): vec/mat copy.
// TODO(lucas): Switch to growable arenas and get rid of individual vec/matrix allocations.
// TODO(lucas): Use scratch space for each pass over the neural network,
// or pre-allocate the memory for each output and have an option for multiplication to take in allocated memory.
// TODO(lucas): Make mat_sretch_* in-place?
// or replace stretching with adding a vector along each axis.
// TODO(lucas): Use shared memory in CUDA code where appropraite (e.g., matrix ops).

int main(void)
{
    rand_seed(123);

    Device device = Device_GPU;

    mat<f32> data;
    vec<u32> labels;
    create_spiral_data(100, 3, &data, &labels, device);

    // 2 input features (x and y coordinates) and 3 output values
    DenseLayer dense1 = dense_layer_init(2, 3, Activation_ReLU, device);
    // Another dense layer
    DenseLayer dense2 = dense_layer_init(3, 3, Activation_ReLU, device);
    // Perform classification using softmax activation to generate a normalized confidence distribution.
    DenseLayer out = dense_layer_init(3, 3, Activation_Softmax, device);

    dense_layer_forward(&dense1, data);
    dense_layer_forward(&dense2, dense1.output);
    dense_layer_forward(&out, dense2.output);

    mat_to(&out.output, Device_CPU);
    mat_print(out.output);

    // TODO(lucas): Add a function to find the max value(s) of a matrix as a whole or along an axis,
    // and the index of the max value.
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

    vec_to(&labels, Device_CPU);
    f32 acc = accuracy_score(labels, pred);

    printf("\nAccuracy: %.2f%%\n", acc*100.0f);

    getchar();
    return 0;
}
