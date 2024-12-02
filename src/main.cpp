#include <stdio.h>

#include "cnnfs.h"

/* TODO(lucas): For project:
 * Add activation function.
 * Final prediction for each data point.
 * Compare each prediction with the labels.
 */

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

    mat<f32> data;
    vec<u8> labels;
    create_spiral_data(100, 3, &data, &labels, DEVICE_CPU);

    // NOTE(lucas): 2 input features (x and y coordinates) and 3 output values
    DenseLayer dense1 = dense_layer_init(2, 3, DEVICE_CPU);
    dense_layer_forward(&dense1, data);
    mat_print(dense1.output);

    getchar();
    return 0;
}
