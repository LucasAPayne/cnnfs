#include "cuda_base.h"
#include "cnnfs_math_cuda.h"
#include "vector_cuda.h"

__global__ internal void linspace_kernel(vec<f32> v, f32 x1, f32 dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    v.data[i] = x1 + ((f32)i*dx);
}

__global__ internal void linspace_kernel(vec<f64> v, f64 x1, f64 dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    v.data[i] = x1 + ((f64)i*dx);
}

__global__ internal void sin_vec_kernel(vec<f32> v, vec<f32> result)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    result.data[i] = sinf(v.data[i]);
}

__global__ internal void sin_vec_kernel(vec<f64> v, vec<f64> result)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    result.data[i] = sin(v.data[i]);
}

__global__ internal void cos_vec_kernel(vec<f32> v, vec<f32> result)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    result.data[i] = cosf(v.data[i]);
}

__global__ internal void cos_vec_kernel(vec<f64> v, vec<f64> result)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    v.data[i] = cos(v.data[i]);
}

__global__ internal void exp_vec_kernel(vec<f32> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    v.data[i] = expf(v.data[i]);
}

__global__ internal void exp_vec_kernel(vec<f64> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    v.data[i] = exp(v.data[i]);
}

__global__ internal void exp_mat_kernel(mat<f32> m)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= m.rows || col >= m.cols)
        return;

    size i = m.cols*row + col;
    m.data[i] = expf(m.data[i]);
}

__global__ internal void exp_mat_kernel(mat<f64> m)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= m.rows || col >= m.cols)
        return;

    size i = m.cols*row + col;
    m.data[i] = exp(m.data[i]);
}

vec<f32> linspace_gpu(f32 x1, f32 x2, size n)
{
    vec<f32> result = vec_zeros_gpu<f32>(n);
    f32 dx = (x2 - x1) / (n - 1.0f);
    linspace_kernel<<<1, 256>>>(result, x1, dx);
    sync_kernel();

    return result;
}

vec<f64> linspace_gpu(f64 x1, f64 x2, size n)
{
    vec<f64> result = vec_zeros_gpu<f64>(n);
    f64 dx = (x2 - x1) / (n - 1.0f);
    linspace_kernel<<<1, 256>>>(result, x1, dx);
    sync_kernel();

    return result;
}

vec<f32> sin_vec_gpu(vec<f32> v)
{
    vec<f32> result = vec_zeros_gpu<f32>(v.elements);
    sin_vec_kernel<<<1, 256>>>(v, result);
    sync_kernel();

    return result;
}

vec<f64> sin_vec_gpu(vec<f64> v)
{
    vec<f64> result = vec_zeros_gpu<f64>(v.elements);
    sin_vec_kernel<<<1, 256>>>(v, result);
    sync_kernel();

    return result;
}

vec<f32> cos_vec_gpu(vec<f32> v)
{
    vec<f32> result = vec_zeros_gpu<f32>(v.elements);
    cos_vec_kernel<<<1, 256>>>(v, result);
    sync_kernel();

    return result;
}

vec<f64> cos_vec_gpu(vec<f64> v)
{
    vec<f64> result = vec_zeros_gpu<f64>(v.elements);
    cos_vec_kernel<<<1, 256>>>(v, result);
    sync_kernel();

    return result;
}

void exp_vec_gpu(vec<f32> v)
{
    exp_vec_kernel<<<1, 256>>>(v);
    sync_kernel();
}

void exp_vec_gpu(vec<f64> v)
{
    exp_vec_kernel<<<1, 256>>>(v);
    sync_kernel();
}

void exp_mat_gpu(mat<f32> m)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    exp_mat_kernel<<<layout.grid_dim, layout.block_dim>>>(m);
    sync_kernel();
}

void exp_mat_gpu(mat<f64> m)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    exp_mat_kernel<<<layout.grid_dim, layout.block_dim>>>(m);
    sync_kernel();
}
