#include "cuda_base.h"
#include "cnnfs_math_cuda.h"
#include "matrix_cuda.h"
#include "vector_cuda.h"

#define SHARED_MEM_SIZE 1024

__global__ internal void linspace_kernel(vec<f32> v, f32 x1, f32 dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] = x1 + ((f32)i*dx);
}

__global__ internal void linspace_kernel(vec<f64> v, f64 x1, f64 dx)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] = x1 + ((f64)i*dx);
}

__global__ internal void sin_vec_kernel(vec<f32> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] = sinf(v.data[i]);
}

__global__ internal void sin_vec_kernel(vec<f64> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] = sin(v.data[i]);
}

__global__ internal void cos_vec_kernel(vec<f32> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] = cosf(v.data[i]);
}

__global__ internal void cos_vec_kernel(vec<f64> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] = cos(v.data[i]);
}

__global__ internal void exp_vec_kernel(vec<f32> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] = expf(v.data[i]);
}

__global__ internal void exp_vec_kernel(vec<f64> v)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i >= v.elements) return;
    
    v.data[i] = exp(v.data[i]);
}

__global__ internal void exp_mat_kernel(mat<f32> m)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    m.data[i] = expf(m.data[i]);
}

__global__ internal void exp_mat_kernel(mat<f64> m)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    m.data[i] = exp(m.data[i]);
}

__device__ void atomicMax(f32* address, f32 value)
{
    if (*address >= value) return;
  
    int* const address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
  
    do
    {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
  
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

__device__ void atomicMax(double* address, double value) {
    if (*address >= value) return;

    unsigned long long int* const address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        if (__longlong_as_double(assumed) >= value) break;

        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

__global__ internal void argmax_kernel(vec<f32> v, f32* max_out, u32* max_idx)
{
    f32 local_max = 0.0f;
    int local_max_idx = 0;

    for (int i = threadIdx.x; i < v.elements; i += blockDim.x)
    {
        f32 val = v.data[i];
        f32 abs_val = abs(val);

        if (local_max < abs_val)
        {
            local_max = abs_val;
            local_max_idx = i;
        }
    }

    atomicMax(max_out, local_max);

    __syncthreads();

    if (*max_out == local_max)
        *max_idx = local_max_idx;
}

// NOTE(lucas): Reference: https://stackoverflow.com/questions/31706599/how-to-perform-reduction-on-a-huge-2d-matrix-along-the-row-direction-using-cuda
__global__ void reduce_block_kernel(mat<f32> m, volatile f32* blk_vals, volatile size* blk_idxs)
{
    __shared__ volatile f32 vals[SHARED_MEM_SIZE];
    __shared__ volatile size idxs[SHARED_MEM_SIZE];
  
    int idx = threadIdx.x+blockDim.x * blockIdx.x;
    int idy = blockIdx.y;
    f32 max_val = -1e10;
    int max_idx = -1;
  
    while (idx < m.cols)
    {
        f32 temp = m.data[idy*m.cols+idx];
        if (temp > max_val)
        {
            max_val = temp;
            max_idx = idx;
        }
        idx += blockDim.x*gridDim.x;
    }

    vals[threadIdx.x] = max_val;
    idxs[threadIdx.x] = max_idx;

    __syncthreads();

    for (int i = (SHARED_MEM_SIZE>>1); i > 0; i>>=1)
    {
        if (threadIdx.x < i)
        {
            if (vals[threadIdx.x] < vals[threadIdx.x + i])
            {
                vals[threadIdx.x] = vals[threadIdx.x+i];
                idxs[threadIdx.x] = idxs[threadIdx.x+i];
            }
        }

      __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        blk_vals[blockIdx.y*m.cols + blockIdx.x] = vals[0];
        blk_idxs[blockIdx.y*m.cols + blockIdx.x] = idxs[0];
    }
}

__global__ void mat_argmax_rows_kernel(volatile f32* blk_vals, volatile size* blk_idxs, size cols, vec<u32> result_max_idx)
{
    __shared__ volatile f32 vals[SHARED_MEM_SIZE];
    __shared__ volatile size idxs[SHARED_MEM_SIZE];
  
    int idx = threadIdx.x;
    int idy = blockIdx.y;
    float my_val = 1e-10;
    int my_idx = -1;
    while (idx < gridDim.x)
    {
        f32 temp = blk_vals[idy*cols + idx];
        if (temp > my_val)
        {
            my_val = temp;
            my_idx = blk_idxs[idy*cols + idx];
        }
        idx += blockDim.x;
    }

    idx = threadIdx.x;
    vals[idx] = my_val;
    idxs[idx] = my_idx;
    __syncthreads();
  
    for (int i = (SHARED_MEM_SIZE>>1); i > 0; i>>=1)
    {
        if (idx < i)
        {
            if (vals[idx] < vals[idx + i])
            {
                vals[idx] = vals[idx+i];
                idxs[idx] = idxs[idx+i];
            }
        }

      __syncthreads();
    }
  
    if(threadIdx.x == 0)
        result_max_idx.data[idy] = idxs[0];
}

vec<f32> linspace_gpu(f32 x1, f32 x2, size n)
{
    vec<f32> result = vec_zeros_gpu<f32>(n);
    f32 dx = (x2 - x1) / (n - 1.0f);
    linspace_kernel<<<1, 256>>>(result, x1, dx);

    return result;
}

vec<f64> linspace_gpu(f64 x1, f64 x2, size n)
{
    vec<f64> result = vec_zeros_gpu<f64>(n);
    f64 dx = (x2 - x1) / (n - 1.0f);
    linspace_kernel<<<1, 256>>>(result, x1, dx);

    return result;
}

void sin_vec_gpu(vec<f32> v)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    sin_vec_kernel<<<layout.grid_dim, layout.block_dim>>>(v);
}

void sin_vec_gpu(vec<f64> v)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    sin_vec_kernel<<<layout.grid_dim, layout.block_dim>>>(v);
}

void cos_vec_gpu(vec<f32> v)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    cos_vec_kernel<<<layout.grid_dim, layout.block_dim>>>(v);
}

void cos_vec_gpu(vec<f64> v)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    cos_vec_kernel<<<layout.grid_dim, layout.block_dim>>>(v);
}

void exp_vec_gpu(vec<f32> v)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    exp_vec_kernel<<<layout.grid_dim, layout.block_dim>>>(v);
}

void exp_vec_gpu(vec<f64> v)
{
    ThreadLayout layout = calc_thread_dim(v.elements);
    exp_vec_kernel<<<layout.grid_dim, layout.block_dim>>>(v);
}

void exp_mat_gpu(mat<f32> m)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    exp_mat_kernel<<<layout.grid_dim, layout.block_dim>>>(m);
}

void exp_mat_gpu(mat<f64> m)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    exp_mat_kernel<<<layout.grid_dim, layout.block_dim>>>(m);
}

u32 argmax_gpu(vec<f32> v)
{
    u32* d_max_idx;
    f32* d_max;
    cuda_call(cudaMalloc(&d_max_idx, sizeof(u32)));
    cuda_call(cudaMalloc(&d_max, sizeof(f32)));

    ThreadLayout layout = calc_thread_dim(v.elements);
    argmax_kernel<<<layout.grid_dim, layout.block_dim>>>(v, d_max, d_max_idx);

    u32 max_idx;
    cuda_call(cudaMemcpy(&max_idx, d_max_idx, sizeof(u32), cudaMemcpyDeviceToHost));
    cuda_call(cudaFree(d_max_idx));
    cuda_call(cudaFree(d_max));

    return max_idx;
}

vec<u32> argmax_gpu(mat<f32> m, Axis axis)
{
    vec<u32> result;
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);

    if (axis != Axis_Rows && axis != Axis_Cols)
        log_invalid_axis(axis);

    // TODO(lucas): For now, column-wise argmax is done by transposing, doing it row-wise, and transposing back.
    // Do an actual reduction along the columns.
    if (axis == Axis_Cols)
        m = transpose_gpu(m);

    result = vec_zeros<u32>(m.rows, Device_GPU);

    f32* blk_vals;
    size* blk_idxs;
    cuda_call(cudaMalloc(&blk_vals, m.rows*layout.block_dim.x*sizeof(f32)));
    cuda_call(cudaMalloc(&blk_idxs, m.rows*layout.block_dim.x*sizeof(size)));

    u32 tpb = SHARED_MEM_SIZE;
    u32 max_blocks_x = (u32)m.cols/tpb + 1;
    dim3 grids(max_blocks_x, (u32)m.rows);
    dim3 threads(tpb, 1);
    dim3 grids2(1, (u32)m.rows);
    dim3 threads2(tpb);

    reduce_block_kernel<<<grids, threads>>>(m, blk_vals, blk_idxs);
    mat_argmax_rows_kernel<<<grids2, threads2>>>(blk_vals, blk_idxs, m.cols, result);

    cuda_call(cudaFree(blk_vals));
    cuda_call(cudaFree(blk_idxs));

    if (axis == Axis_Cols)
        m = transpose_gpu(m);

    return result;
}

// #define VEC_ARGMAX_GPU(T) INST_TEMPLATE(argmax_gpu, u32, T, (vec<T> v))
// #define MAT_ARGMAX_GPU(T) INST_TEMPLATE(argmax_gpu, vec<u32>, T, (mat<T> m, Axis axis))

// INST_ALL_TYPES(VEC_ARGMAX_GPU)
// INST_ALL_TYPES(MAT_ARGMAX_GPU)

// #undef VEC_ARGMAX_GPU
// #undef MAT_ARGMAX_GPU
