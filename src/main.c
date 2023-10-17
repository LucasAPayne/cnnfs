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
    create_spiral_data(100, 3, &data, &labels);

    dense_layer_f32 dense1 = dense_layer_f32_init(2, 3);
    dense_layer_f32_forward(&dense1, data);
    mat_f32_print(dense1.output);

    getchar();

    return 0;
}
