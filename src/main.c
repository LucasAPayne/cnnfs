#include "datasets.h"
#include "dense_layer.h"
#include "math/cnnfs_math.h"
#include "util/rng.h"
#include "util/types.h"

#include <stdio.h> // getchar

int main()
{
    rand_seed(123);

    mat_f32 data;
    vec_u8 labels;
    create_spiral_data(10, 3, &data, &labels);

    f32 weight_data[] =
    {
        -0.01306527f,  0.01658131f, -0.00118164f,
        -0.00680178f,  0.00666383f, -0.0046072f
    };
    mat_f32 weights = mat_f32_init(2, 3, weight_data);
    mat_f32 biases = mat_f32_zeros(1, 3);
    mat_f32 outputs = mat_f32_zeros(2, 3);

    dense_layer dense1 = {0};
    dense1.weights = weights;
    dense1.biases = biases;
    dense1.output = outputs;

    dense_layer_forward(&dense1, data);

    mat_f32_print(dense1.output);

    getchar();

    return 0;
}
