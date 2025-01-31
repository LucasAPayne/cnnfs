#include "dense_layer.h"
#include "util/rng.h"

DenseLayer dense_layer_init(size inputs, size neurons, ActivationType activation, Device device)
{
    DenseLayer layer = {0};

    layer.weights = mat_rand_gauss_standard(inputs, neurons);
    mat_to(&layer.weights, device);
    layer.weights *= 0.01f;
    layer.biases = mat_zeros<f32>(1, neurons, device);
    layer.activation = activation;

    return layer;
}

void dense_layer_forward(DenseLayer* layer, mat<f32> input)
{
    layer->output = input*layer->weights;
    mat<f32> temp = mat_stretch_add(layer->output, layer->biases);
    mat_free_data(layer->output);
    layer->output = temp;
    activation_forward(layer->output, layer->activation);
}
