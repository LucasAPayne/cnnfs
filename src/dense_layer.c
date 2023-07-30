#include "dense_layer.h"

void dense_layer_forward(dense_layer* layer, mat_f32 input)
{
    layer->output = mat_f32_mul(input, layer->weights);
    layer->output = mat_f32_add(layer->output, layer->biases);
}
