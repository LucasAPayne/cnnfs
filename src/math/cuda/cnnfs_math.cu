#include "cuda_base.h"
#include "cnnfs_math_cuda.h"

__global__ internal void linspace_kernel(vec<f32> v, f32 x1, f32 dx)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= v.elements)
        return;
    
    v.data[idx] = x1 + ((f32)idx*dx);
}

__global__ internal void linspace_kernel(vec<f64> v, f64 x1, f64 dx)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= v.elements)
        return;
    
    v.data[idx] = x1 + ((f64)idx*dx);
}

__global__ internal void sin_vec_kernel(vec<f32> v, vec<f32> result)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= v.elements)
        return;
    
    result.data[idx] = sinf(v.data[idx]);
}

__global__ internal void sin_vec_kernel(vec<f64> v, vec<f64> result)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= v.elements)
        return;
    
    result.data[idx] = sin(v.data[idx]);
}

__global__ internal void cos_vec_kernel(vec<f32> v, vec<f32> result)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= v.elements)
        return;
    
    result.data[idx] = cosf(v.data[idx]);
}

__global__ internal void cos_vec_kernel(vec<f64> v, vec<f64> result)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= v.elements)
        return;
    
    v.data[idx] = cos(v.data[idx]);
}

vec<f32> linspace_gpu(f32 x1, f32 x2, usize n)
{
    vec<f32> result = vec_zeros<f32>(n, DEVICE_GPU);
    f32 dx = (x2 - x1) / (n - 1.0f);
    linspace_kernel<<<1, 256>>>(result, x1, dx);
    sync_kernel();

    return result;
}

vec<f64> linspace_gpu(f64 x1, f64 x2, usize n)
{
    vec<f64> result = vec_zeros<f64>(n, DEVICE_GPU);
    f64 dx = (x2 - x1) / (n - 1.0f);
    linspace_kernel<<<1, 256>>>(result, x1, dx);
    sync_kernel();

    return result;
}

vec<f32> sin_vec_gpu(vec<f32> v)
{
    vec<f32> result = vec_zeros<f32>(v.elements);
    sin_vec_kernel<<<1, 256>>>(v, result);
    sync_kernel();

    return result;
}

vec<f64> sin_vec_gpu(vec<f64> v)
{
    vec<f64> result = vec_zeros<f64>(v.elements);
    sin_vec_kernel<<<1, 256>>>(v, result);
    sync_kernel();

    return result;
}

vec<f32> cos_vec_gpu(vec<f32> v)
{
    vec<f32> result = vec_zeros<f32>(v.elements);
    cos_vec_kernel<<<1, 256>>>(v, result);
    sync_kernel();

    return result;
}

vec<f64> cos_vec_gpu(vec<f64> v)
{
    vec<f64> result = vec_zeros<f64>(v.elements);
    cos_vec_kernel<<<1, 256>>>(v, result);
    sync_kernel();

    return result;
}
