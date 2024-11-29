#include "matrix.h"

// host code for validating last cuda operation (not kernel launch)
#define cuda_call(ans) { gpu_assert((ans), __FILE__, __LINE__); }
void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        getchar();
        if (abort)
            exit(code);
    }
}

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

mat_f32 mat_f32_to(mat_f32 a, Device device)
{
    mat_f32 result = {};
    result.rows = a.rows;
    result.cols = a.cols;

    switch (a.device)
    {
        case DEVICE_CPU:
        {
            if (device == DEVICE_GPU)
            {
                usize mem = a.rows*a.cols*sizeof(f32);
                cuda_call(cudaMalloc(&result.data, mem));
                cuda_call(cudaMemcpy(result.data, a.data, mem, cudaMemcpyHostToDevice));
                result.device = DEVICE_GPU;
            }
        } break;

        case DEVICE_GPU:
        {
            if (device == DEVICE_CPU)
            {
                usize mem = a.rows*a.cols*sizeof(f32);
                result.data = (f32*)malloc(mem);
                cuda_call(cudaMemcpy(result.data, a.data, mem, cudaMemcpyDeviceToHost));
                result.device = DEVICE_CPU;
            }
        } break;
        default: break; // TODO(lucas): Log invalid device error.
    }

    return result;
}

mat_f32 mat_f32_zeros_gpu(usize rows, usize cols)
{
    mat_f32 result = {};
    result.rows = rows;
    result.cols = cols;
    result.device = DEVICE_GPU;

    usize mem = rows*cols*sizeof(f32);
    cuda_call(cudaMalloc(&result.data, mem));
    cuda_call(cudaMemset(result.data, 0, mem));

    return result;
}

void mat_f32_add_gpu(mat_f32 a, mat_f32 b, mat_f32 c)
{
    dim3 blockDim(16, 16); // 16x16 threads per block
    dim3 gridDim(((u32)a.cols + blockDim.x - 1) / blockDim.x,
                    ((u32)a.rows + blockDim.y - 1) / blockDim.y);

    mat_f32_add_kernel<<<gridDim, blockDim>>>(a.data, b.data, c.data, a.rows, a.cols);
    cudaDeviceSynchronize();
}

}
