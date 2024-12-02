#include "cuda_base.h"
#include "vector_cuda.h"

template <typename T>
__global__ internal void vec_full_kernel(vec<T> v, T fill_value)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= v.elements)
        return;
    
    v.data[idx] = fill_value;
}

template <typename T>
__global__ internal void vec_set_range_kernel(vec<T> v, vec<T> data, usize offset)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= data.elements)
        return;
    
    v.data[offset + idx] = data.data[idx];
}

template <typename T>
__global__ internal void vec_add_kernel(vec<T> a, vec<T> b)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= a.elements)
        return;
    
    a.data[idx] += b.data[idx];
}

template <typename T>
__global__ internal void vec_scale_kernel(vec<T> v, T c)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= v.elements)
        return;
    
    v.data[idx] *= c;
}

template <typename T>
__global__ internal void vec_had_kernel(vec<T> a, vec<T> b)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx >= a.elements)
        return;
    
    a.data[idx] *= b.data[idx];
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
    vec<T> result = vec_zeros_gpu<T>(elements);
    vec_full_kernel<<<1, 256>>>(result, fill_value);
    sync_kernel();

    return result;
}

template <typename T>
void vec_set_range_gpu(vec<T> v, vec<T> data, usize offset)
{
    vec_set_range_kernel<<<1, 256>>>(v, data, offset);
    sync_kernel();
}

template <typename T>
void vec_add_gpu(vec<T> a, vec<T> b)
{
    vec_add_kernel<<<1, 256>>>(a, b);
    sync_kernel();
}

template <typename T>
void vec_scale_gpu(vec<T> v, T c)
{
    vec_scale_kernel<<<1, 256>>>(v, c);
    sync_kernel();
}

template <typename T>
void vec_had_gpu(vec<T> a, vec<T> b)
{
    vec_had_kernel<<<1, 256>>>(a, b);
    sync_kernel();
}

#define VEC_TO(T) INST_TEMPLATE(vec_to, void, T, (vec<T>* v, Device device))
#define VEC_ZEROS_GPU(T) INST_TEMPLATE(vec_zeros_gpu, vec<T>, T, (usize elements))
#define VEC_FULL_GPU(T) INST_TEMPLATE(vec_full_gpu, vec<T>, T, (usize elements, T fill_value))
#define VEC_SET_RANGE_GPU(T) INST_TEMPLATE(vec_set_range_gpu, void, T, (vec<T> v, vec<T> data, usize offset))
#define VEC_ADD_GPU(T) INST_TEMPLATE(vec_add_gpu, void, T, (vec<T> a, vec<T> b))
#define VEC_SCALE_GPU(T) INST_TEMPLATE(vec_scale_gpu, void, T, (vec<T> v, T c))
#define VEC_HAD_GPU(T) INST_TEMPLATE(vec_had_gpu, void, T, (vec<T> a, vec<T> b))

INST_ALL_TYPES(VEC_TO)
INST_ALL_TYPES(VEC_ZEROS_GPU)
INST_ALL_TYPES(VEC_FULL_GPU)
INST_ALL_TYPES(VEC_SET_RANGE_GPU)
INST_ALL_TYPES(VEC_ADD_GPU)
INST_ALL_TYPES(VEC_SCALE_GPU)
INST_ALL_TYPES(VEC_HAD_GPU)

#undef VEC_TO
#undef VEC_ZEROS_GPU
#undef VEC_FULL_GPU
#undef VEC_SET_RANGE_GPU
#undef VEC_ADD_GPU
#undef VEC_SCALE_GPU
#undef VEC_HAD_GPU
