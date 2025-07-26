#include "math/cuda/cnnfs_activation_cuda.h"
#include "cuda_base.h"
#include "cnnfs_math_cuda.h"
#include "vector_cuda.h"

__global__ internal void relu_forward_kernel(mat<f32> inputs)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= inputs.rows || col >= inputs.cols) return;

    size idx = inputs.cols*row + col;
    inputs.data[idx] = max(0.0f, inputs.data[idx]);
}

__global__ internal void relu_forward_kernel(mat<f64> inputs)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= inputs.rows || col >= inputs.cols)
        return;

    size idx = inputs.cols*row + col;
    inputs.data[idx] = max(0.0, inputs.data[idx]);
}

void relu_forward_gpu(mat<f32> inputs)
{
    ThreadLayout layout = calc_thread_dim(inputs.rows, inputs.cols);
    relu_forward_kernel<<<layout.grid_dim, layout.block_dim>>>(inputs);
    cuda_call(cudaGetLastError());
}

void relu_forward_gpu(mat<f64> inputs)
{
    ThreadLayout layout = calc_thread_dim(inputs.rows, inputs.cols);
    relu_forward_kernel<<<layout.grid_dim, layout.block_dim>>>(inputs);
    cuda_call(cudaGetLastError());
}
