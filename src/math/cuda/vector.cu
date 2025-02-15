#include "cuda_base.h"
#include "vector_cuda.h"
#include "profile.h"
#include "util/rng.h"

#include <curand_kernel.h>

template <typename T>
__global__ internal void vec_full_kernel(vec<T> result, T fill_value)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= result.elements) return;
    
    result.data[i] = fill_value;
}

__global__ internal void vec_rand_uniform_kernel(vec<f32> result, f32 min, f32 max, u64 seed)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= result.elements) return;

    curandState state = {};
    curand_init(seed+i, 0, 0, &state);
    f32 val = curand_uniform(&state);
    result.data[i] = min + val * (max - min);
}

__global__ internal void vec_rand_gauss_kernel(vec<f32> result, f32 mean, f32 std_dev, u64 seed)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (2*i > result.elements) return;

    curandState state = {};
    curand_init(seed+i, 0, 0, &state);
    float2 vals = curand_normal2(&state);

    // NOTE(lucas): curand_normal uses a standard Gaussian distribution,
    // so it needs to be shifted to the desired mean and standard deviation.
    result.data[2*i] = vals.x*std_dev + mean;

    if (2*i + 1 >= result.elements) return;
    result.data[2*i+1] = vals.y*std_dev + mean;
}

__global__ internal void vec_rand_gauss_standard_kernel(vec<f32> result, u64 seed)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (2*i > result.elements) return;

    curandState state = {};
    curand_init(seed+i, 0, 0, &state);
    float2 vals = curand_normal2(&state);

    result.data[2*i] = vals.x;
    if (2*i + 1 >= result.elements) return;
    result.data[2*i + 1] = vals.y;
}

template <typename T>
__global__ internal void vec_copy_kernel(vec<T> result, vec<T> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    result.data[i] = v.data[i];
}

template <typename T>
__global__ internal void vec_set_range_kernel(vec<T> v, vec<T> data, size offset)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= data.elements) return;
    
    v.data[offset + i] = data.data[i];
}

template <typename T>
__global__ internal void vec_add_kernel(vec<T> a, vec<T> b)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= a.elements) return;
    
    a.data[i] += b.data[i];
}

template <typename T>
__global__ internal void vec_scale_kernel(vec<T> v, T c)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] *= c;
}

template <typename T>
__global__ internal void vec_scale_inv_kernel(vec<T> v, T c)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    if (v.data[i] != 0)
        v.data[i] = (T)c / v.data[i];
}

template <typename T>
__global__ internal void vec_had_kernel(vec<T> a, vec<T> b)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= a.elements) return;
    
    a.data[i] *= b.data[i];
}

template <typename T>
__global__ internal void vec_sum_kernel(vec<T> v, T* out)
{
    __shared__ T shared_mem[256];

    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // Each thread loads one element from global memory to shared memory.
    // Zero-pad shared memory if the size of the array is not an exact multiple of shared memory size.
    if (i < v.elements)
        shared_mem[threadIdx.x] = v.data[i];
    else
        shared_mem[threadIdx.x] = (T)0;

    __syncthreads();

    // Perform reduction.
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int idx = 2*s*threadIdx.x;

        if (idx < blockDim.x)
            shared_mem[idx] += shared_mem[idx + s];
    }

    __syncthreads();

    // Write block result to global memory.
    if (threadIdx.x == 0)
        out[blockIdx.x] = shared_mem[0];
}

template <typename T>
void vec_to(vec<T>* v, Device device)
{
    vec<T> result = {};
    result.elements = v->elements;
    switch (v->device)
    {
        case Device_CPU:
        {
            if (device == Device_GPU)
            {
                size mem = v->elements*sizeof(T);
                cuda_call(cudaMalloc(&result.data, mem));
                cuda_call(cudaMemcpy(result.data, v->data, mem, cudaMemcpyHostToDevice));
                result.device = Device_GPU;
                free(v->data);
                *v = result;
            }
        } break;

        case Device_GPU:
        {
            if (device == Device_CPU)
            {
                size mem = v->elements*sizeof(T);
                result.data = (T*)malloc(mem);
                cuda_call(cudaMemcpy(result.data, v->data, mem, cudaMemcpyDeviceToHost));
                result.device = Device_CPU;
                cuda_call(cudaFree(v->data));
                *v = result;
            }
        } break;

        default: log_invalid_device(device); break;
    }
}

template <typename T>
vec<T> vec_zeros_gpu(size elements)
{
    vec<T> result = {};
    result.elements = elements;
    result.device = Device_GPU;

    size mem = elements*sizeof(T);
    cuda_call(cudaMalloc(&result.data, mem));
    cuda_call(cudaMemset(result.data, 0, mem));

    ASSERT(result.data, "Vector GPU allocation failed.\n");

    return result;
}

template <typename T>
vec<T> vec_full_gpu(size elements, T fill_value)
{
    ThreadLayout layout = calc_thread_dim(elements);
    vec<T> result = vec_zeros_gpu<T>(elements);
    vec_full_kernel<<<layout.grid_dim, layout.block_dim>>>(result, fill_value);

    return result;
}

vec<f32> vec_rand_uniform_gpu(size n, f32 min, f32 max)
{
    vec<f32> result = vec_zeros_gpu<f32>(n);
    ThreadLayout layout = calc_thread_dim(n);
    vec_rand_uniform_kernel<<<layout.grid_dim, layout.block_dim>>>(result, min, max, gpu_rng_global.seed);

    return result;
}

vec<f32> vec_rand_gauss_gpu(size n, f32 mean, f32 std_dev)
{
    vec<f32> result = vec_zeros_gpu<f32>(n);
    ThreadLayout layout = calc_thread_dim(n);
    vec_rand_gauss_kernel<<<layout.grid_dim, layout.block_dim>>>(result, mean, std_dev, gpu_rng_global.seed);

    return result;
}

vec<f32> vec_rand_gauss_standard_gpu(size n)
{
    vec<f32> result = vec_zeros_gpu<f32>(n);
    ThreadLayout layout = calc_thread_dim(n);
    vec_rand_gauss_standard_kernel<<<layout.grid_dim, layout.block_dim>>>(result, gpu_rng_global.seed);

    return result;
}

template <typename T>
vec<T> vec_copy_gpu(vec<T> v)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec<T> result = vec_zeros_gpu<T>(v.elements);
    vec_copy_kernel<<<layout.grid_dim, layout.block_dim>>>(result, v);

    return result;
}

template <typename T>
void vec_set_range_gpu(vec<T> v, vec<T> data, size offset)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_set_range_kernel<<<layout.grid_dim, layout.block_dim>>>(v, data, offset);
}

template <typename T>
void vec_add_gpu(vec<T> a, vec<T> b)
{
    ThreadLayout layout = calc_thread_dim(a.elements);
    vec_add_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b);
}

template <typename T>
void vec_scale_gpu(vec<T> v, T c)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_scale_kernel<<<layout.grid_dim, layout.block_dim>>>(v, c);
}

template <typename T>
vec<T> vec_scale_inv_gpu(vec<T> v, T c)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_scale_inv_kernel<<<layout.grid_dim, layout.block_dim>>>(v, c);
    return v;
}

template <typename T>
void vec_had_gpu(vec<T> a, vec<T> b)
{
    ThreadLayout layout = calc_thread_dim(a.elements);
    vec_had_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b);
}

template <typename T>
T vec_sum_gpu(vec<T> v)
{
    T* out;
    size mem = v.elements*sizeof(T);
    cuda_call(cudaMalloc(&out, mem));

    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_sum_kernel<<<layout.grid_dim, layout.block_dim>>>(v, out);
    
    T result;
    cuda_call(cudaMemcpy(&result, out, sizeof(T), cudaMemcpyDeviceToHost));
    cuda_call(cudaFree(out));

    return result;
}

#define VEC_TO(T) INST_TEMPLATE(vec_to, void, T, (vec<T>* v, Device device))
#define VEC_ZEROS_GPU(T) INST_TEMPLATE(vec_zeros_gpu, vec<T>, T, (size elements))
#define VEC_FULL_GPU(T) INST_TEMPLATE(vec_full_gpu, vec<T>, T, (size elements, T fill_value))
#define VEC_COPY_GPU(T) INST_TEMPLATE(vec_copy_gpu, vec<T>, T, (vec<T> v))
#define VEC_SET_RANGE_GPU(T) INST_TEMPLATE(vec_set_range_gpu, void, T, (vec<T> v, vec<T> data, size offset))
#define VEC_ADD_GPU(T) INST_TEMPLATE(vec_add_gpu, void, T, (vec<T> a, vec<T> b))
#define VEC_SCALE_GPU(T) INST_TEMPLATE(vec_scale_gpu, void, T, (vec<T> v, T c))
#define VEC_SCALE_INV_GPU(T) INST_TEMPLATE(vec_scale_inv_gpu, vec<T>, T, (vec<T> v, T c))
#define VEC_HAD_GPU(T) INST_TEMPLATE(vec_had_gpu, void, T, (vec<T> a, vec<T> b))
#define VEC_SUM_GPU(T) INST_TEMPLATE(vec_sum_gpu, T, T, (vec<T> v))

INST_ALL_TYPES(VEC_TO)
INST_ALL_TYPES(VEC_ZEROS_GPU)
INST_ALL_TYPES(VEC_FULL_GPU)
INST_ALL_TYPES(VEC_COPY_GPU)
INST_ALL_TYPES(VEC_SET_RANGE_GPU)
INST_ALL_TYPES(VEC_ADD_GPU)
INST_ALL_TYPES(VEC_SCALE_GPU)
INST_ALL_TYPES(VEC_SCALE_INV_GPU)
INST_ALL_TYPES(VEC_HAD_GPU)
INST_ALL_TYPES(VEC_SUM_GPU)

#undef VEC_TO
#undef VEC_ZEROS_GPU
#undef VEC_FULL_GPU
#undef VEC_COPY_GPU
#undef VEC_SET_RANGE_GPU
#undef VEC_ADD_GPU
#undef VEC_SCALE_GPU
#undef VEC_SCALE_INV_GPU
#undef VEC_HAD_GPU
#undef VEC_SUM_GPU
