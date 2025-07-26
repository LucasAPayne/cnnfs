#include "metrics_cuda.h"
#include "cuda_base.h"

__global__ internal void accuracy_score_kernel(u32* d_true, u32* d_pred, u32* result, size n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;

    u32 correct = (u32)(d_true[i] == d_pred[i]);
    atomicAdd(result, correct);
}

f32 accuracy_score_gpu(vec<u32> y_true, vec<u32> y_pred)
{
    u32* d_sum;
    size mem = y_true.elements*sizeof(u32);
    cuda_call(cudaMalloc(&d_sum, mem));
    cuda_call(cudaMemset(d_sum, 0, sizeof(u32)));

    u32* d_true;
    u32* d_pred;
    cuda_call(cudaMalloc(&d_true, mem));
    cuda_call(cudaMalloc(&d_pred, mem));
    cuda_call(cudaMemcpy(d_true, y_true.data, mem, cudaMemcpyHostToDevice));
    cuda_call(cudaMemcpy(d_pred, y_pred.data, mem, cudaMemcpyHostToDevice));

    ThreadLayout layout = calc_thread_dim(y_true.elements);
    accuracy_score_kernel<<<layout.grid_dim, layout.block_dim>>>(d_true, d_pred, d_sum, y_true.elements);
    cuda_call(cudaGetLastError());

    u32 h_sum = 0;
    cuda_call(cudaMemcpy(&h_sum, d_sum, sizeof(u32), cudaMemcpyDeviceToHost));
    cuda_call(cudaFree(d_sum));
    cuda_call(cudaFree(d_true));
    cuda_call(cudaFree(d_pred));

    f32 acc = (f32)h_sum / (f32)y_true.elements;
    return acc;
}
