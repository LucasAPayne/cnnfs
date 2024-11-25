#include "matrix.h"

__global__ void mat_f32_add_kernel(f32* a, f32* b, f32* out, usize rows, usize cols)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if (x >= cols || y >= rows)
        return;
    
    usize idx = y*cols + x;
    out[idx] = a[idx] + b[idx];
}

extern "C" {

void mat_f32_add_gpu(mat_f32 a, mat_f32 b, mat_f32* c)
{
    // TODO(lucas): Should already be on the GPU at this point.
    mat_f32 a_gpu;
    mat_f32 b_gpu;
    mat_f32 c_gpu;

    usize size = a.rows*a.cols*sizeof(f32);
    cudaMalloc(&a_gpu.data, size);
    cudaMalloc(&b_gpu.data, size);
    cudaMalloc(&c_gpu.data, size);

    cudaMemcpy(a_gpu.data, a.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_gpu.data, b.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(c_gpu.data, c->data, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim((a.cols + blockDim.x - 1) / blockDim.x,
                    (a.rows + blockDim.y - 1) / blockDim.y);

    mat_f32_add_kernel<<<gridDim, blockDim>>>(a_gpu.data, b_gpu.data, c_gpu.data, a.rows, a.cols);

    cudaMemcpy(c->data, c_gpu.data, size, cudaMemcpyDeviceToHost);
}

}
