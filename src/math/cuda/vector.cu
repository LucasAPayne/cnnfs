#include "cuda_base.h"
#include "vector_cuda.h"

template <typename T>
__global__ internal void vec_full_kernel(vec<T> v, T fill_value)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    v.data[i] = fill_value;
}

template <typename T>
__global__ internal void vec_set_range_kernel(vec<T> v, vec<T> data, usize offset)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= data.elements)
        return;
    
    v.data[offset + i] = data.data[i];
}

template <typename T>
__global__ internal void vec_add_kernel(vec<T> a, vec<T> b)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= a.elements)
        return;
    
    a.data[i] += b.data[i];
}

template <typename T>
__global__ internal void vec_scale_kernel(vec<T> v, T c)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    v.data[i] *= c;
}

template <typename T>
__global__ internal void vec_reciprocal_kernel(vec<T> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= v.elements)
        return;
    
    if (v.data[i])
        v.data[i] = (T)1 / v.data[i];
}

template <typename T>
__global__ internal void vec_had_kernel(vec<T> a, vec<T> b)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;

    if (i >= a.elements)
        return;
    
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
        shared_mem[threadIdx.x] = 0;

    __syncthreads();

    // Perform reduction.
    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int idx = 2*s*threadIdx.x;

        if (idx < blockDim.x)
            shared_mem[idx] += shared_mem[idx + s];
        
        __syncthreads();
    }

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
        case DEVICE_CPU:
        {
            if (device == DEVICE_GPU)
            {
                usize mem = v->elements*sizeof(T);
                cuda_call(cudaMalloc(&result.data, mem));
                cuda_call(cudaMemcpy(result.data, v->data, mem, cudaMemcpyHostToDevice));
                result.device = DEVICE_GPU;
                free(v->data);
                *v = result;
            }
        } break;

        case DEVICE_GPU:
        {
            if (device == DEVICE_CPU)
            {
                usize mem = v->elements*sizeof(T);
                result.data = (T*)malloc(mem);
                cuda_call(cudaMemcpy(result.data, v->data, mem, cudaMemcpyDeviceToHost));
                result.device = DEVICE_CPU;
                cuda_call(cudaFree(v->data));
                *v = result;
            }
        }

        default: break;
    }
}

template <typename T>
vec<T> vec_zeros_gpu(usize elements)
{
    vec<T> result = {};
    result.elements = elements;
    result.device = DEVICE_GPU;

    usize mem = elements*sizeof(T);
    cuda_call(cudaMalloc(&result.data, mem));
    cuda_call(cudaMemset(result.data, 0, mem));

    return result;
}

template <typename T>
vec<T> vec_full_gpu(usize elements, T fill_value)
{
    ThreadLayout layout = calc_thread_dim(elements);
    vec<T> result = vec_zeros_gpu<T>(elements);
    vec_full_kernel<<<layout.grid_dim, layout.block_dim>>>(result, fill_value);
    sync_kernel();

    return result;
}

template <typename T>
void vec_set_range_gpu(vec<T> v, vec<T> data, usize offset)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_set_range_kernel<<<layout.grid_dim, layout.block_dim>>>(v, data, offset);
    sync_kernel();
}

template <typename T>
void vec_add_gpu(vec<T> a, vec<T> b)
{
    ThreadLayout layout = calc_thread_dim(a.elements);
    vec_add_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b);
    sync_kernel();
}

template <typename T>
void vec_scale_gpu(vec<T> v, T c)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_scale_kernel<<<layout.grid_dim, layout.block_dim>>>(v, c);
    sync_kernel();
}

template <typename T>
vec<T> vec_reciprocal_gpu(vec<T> v)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_reciprocal_kernel<<<layout.grid_dim, layout.block_dim>>>(v);
    sync_kernel();
    return v;
}

template <typename T>
void vec_had_gpu(vec<T> a, vec<T> b)
{
    ThreadLayout layout = calc_thread_dim(a.elements);
    vec_had_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b);
    sync_kernel();
}

// TODO(lucas): Temporary, will convert to template after testing other types.
f32 vec_sum_gpu(vec<f32> v)
{
    f32* out;
    usize mem = v.elements*sizeof(f32);
    cuda_call(cudaMalloc(&out, mem));
    
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_sum_kernel<<<layout.grid_dim, layout.block_dim>>>(v, out);
    sync_kernel();

    f32 result;
    cuda_call(cudaMemcpy(&result, out, sizeof(f32), cudaMemcpyDeviceToHost));
    cuda_call(cudaFree(out));
    
    return result;
}

f64 vec_sum_gpu(vec<f64> v)
{
    f64* out;
    usize mem = v.elements*sizeof(f64);
    cuda_call(cudaMalloc(&out, mem));
    
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_sum_kernel<<<layout.grid_dim, layout.block_dim>>>(v, out);
    sync_kernel();

    f64 result;
    cuda_call(cudaMemcpy(&result, out, sizeof(f32), cudaMemcpyDeviceToHost));
    cuda_call(cudaFree(out));
    
    return result;
}

u32 vec_sum_gpu(vec<u32> v)
{
    u32* out;
    usize mem = v.elements*sizeof(u32);
    cuda_call(cudaMalloc(&out, mem));
    
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_sum_kernel<<<layout.grid_dim, layout.block_dim>>>(v, out);
    sync_kernel();

    u32 result;
    cuda_call(cudaMemcpy(&result, out, sizeof(f32), cudaMemcpyDeviceToHost));
    cuda_call(cudaFree(out));
    
    return result;
}

i32 vec_sum_gpu(vec<i32> v)
{
    i32* out;
    usize mem = v.elements*sizeof(i32);
    cuda_call(cudaMalloc(&out, mem));
    
    ThreadLayout layout = calc_thread_dim(v.elements);
    vec_sum_kernel<<<layout.grid_dim, layout.block_dim>>>(v, out);
    sync_kernel();

    i32 result;
    cuda_call(cudaMemcpy(&result, out, sizeof(f32), cudaMemcpyDeviceToDevice));
    cuda_call(cudaFree(out));
    
    return result;
}

#define VEC_TO(T) INST_TEMPLATE(vec_to, void, T, (vec<T>* v, Device device))
#define VEC_ZEROS_GPU(T) INST_TEMPLATE(vec_zeros_gpu, vec<T>, T, (usize elements))
#define VEC_FULL_GPU(T) INST_TEMPLATE(vec_full_gpu, vec<T>, T, (usize elements, T fill_value))
#define VEC_SET_RANGE_GPU(T) INST_TEMPLATE(vec_set_range_gpu, void, T, (vec<T> v, vec<T> data, usize offset))
#define VEC_ADD_GPU(T) INST_TEMPLATE(vec_add_gpu, void, T, (vec<T> a, vec<T> b))
#define VEC_SCALE_GPU(T) INST_TEMPLATE(vec_scale_gpu, void, T, (vec<T> v, T c))
#define VEC_RECIPROCAL_GPU(T) INST_TEMPLATE(vec_reciprocal_gpu, vec<T>, T, (vec<T> v))
#define VEC_HAD_GPU(T) INST_TEMPLATE(vec_had_gpu, void, T, (vec<T> a, vec<T> b))

INST_ALL_TYPES(VEC_TO)
INST_ALL_TYPES(VEC_ZEROS_GPU)
INST_ALL_TYPES(VEC_FULL_GPU)
INST_ALL_TYPES(VEC_SET_RANGE_GPU)
INST_ALL_TYPES(VEC_ADD_GPU)
INST_ALL_TYPES(VEC_SCALE_GPU)
INST_ALL_TYPES(VEC_RECIPROCAL_GPU)
INST_ALL_TYPES(VEC_HAD_GPU)

#undef VEC_TO
#undef VEC_ZEROS_GPU
#undef VEC_FULL_GPU
#undef VEC_SET_RANGE_GPU
#undef VEC_ADD_GPU
#undef VEC_SCALE_GPU
#undef VEC_RECIPROCAL_GPU
#undef VEC_HAD_GPU
