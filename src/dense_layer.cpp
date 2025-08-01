#include "dense_layer.h"
#include "util/rng.h"

DenseLayer dense_layer_init(size inputs, size neurons, ActivationType activation, Device device)
{
    DenseLayer layer = {0};

    layer.weights = mat_rand_gauss_standard(inputs, neurons, device);
    layer.weights *= 0.01f;
    layer.biases = vec_zeros<f32>(neurons, device);
    layer.activation = activation;

    return layer;
}

void dense_layer_forward(DenseLayer* layer, mat<f32> input)
{
    layer->output = input*layer->weights;
    mat_add_vec(layer->output, layer->biases);
    activation_forward(layer->output, layer->activation);
}
