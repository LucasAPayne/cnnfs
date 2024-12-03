#include <stdio.h>

#include "cnnfs.h"

/* TODO(lucas): For project:
 * Final prediction for each data point.
 * Compare each prediction with the labels.
 */

// TODO(lucas): Get rid of atomic_add functions and fix matrix and vector sums.
// TODO(lucas): Change capitalization to Device_CPU, Device_GPU for consistency.
// TODO(lucas): Matrix and vector functions should still operate in-place, but also return the value for more expressiveness.
// TODO(lucas): Try to eliminate copying data between host and device in mat_sum_gpu and mat_scale_gpu
// TODO(lucas): Look into another way of specifying that some function templates are only valid for certain types,
// besides just making multiple copies, to make for easier maintenance.
// TODO(lucas): Some math and other functions should have an option of operating in-place or returning a copy.
// TODO(lucas): Should the math functions just be rolled into vector.h?
// TODO(lucas): Drop mat/vec prefixes for functions and just use overloading.
// TODO(lucas): Add operator overloading.
// TODO(lucas): Add RNG for GPU.
// TODO(lucas): Add profiling.
// TODO(lucas): Add logging.
// TODO(lucas): Log invalid device errors and other assertions.
// TODO(lucas): vec/mat copy.
// TODO(lucas): Switch to growable arenas and get rid of individual vec/matrix allocations.
// TODO(lucas): Use scratch space for each pass over the neural network.
// TODO(lucas): Make mat_sretch_* in-place?
// TODO(lucas): Use shared memory in CUDA code where appropraite (e.g., matrix ops).

int main(void)
{
    rand_seed(123);

    Device device = DEVICE_GPU;

    mat<f32> data;
    vec<u32> labels;
    create_spiral_data(100, 3, &data, &labels, device);

    // NOTE(lucas): 2 input features (x and y coordinates) and 3 output values
    DenseLayer dense1 = dense_layer_init(2, 3, Activation_ReLU, device);
    // NOTE(lucas): Perform classification using softmax activation to generate a normalized confidence distribution.
    DenseLayer dense2 = dense_layer_init(3, 3, Activation_Softmax, device);

    dense_layer_forward(&dense1, data);
    dense_layer_forward(&dense2, dense1.output);

    mat_print(dense2.output);

    getchar();
    return 0;
}
