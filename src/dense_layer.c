#include "dense_layer.h"
#include "util/rng.h"

dense_layer_f32 dense_layer_f32_init(usize inputs, usize neurons)
{
    dense_layer_f32 layer = {0};

    layer.weights = mat_f32_scale(rand_mat_f32_gauss_standard(inputs, neurons), 0.01f);
    layer.biases = mat_f32_zeros(1, neurons);

    return layer;
}

void dense_layer_f32_forward(dense_layer_f32* layer, mat_f32 input)
{
    layer->output = mat_f32_mul(input, layer->weights);
    layer->output = mat_f32_add(layer->output, layer->biases);
}
