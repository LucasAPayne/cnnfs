#include "cuda_base.h"
#include "matrix_cuda.h"
#include "vector_cuda.h"
#include "util/rng.h"

#include <curand_kernel.h>

template <typename T>
__global__ internal void mat_full_kernel(mat<T> result, T fill_value)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= result.rows || col >= result.cols) return;

    size i = row*result.cols + col;
    result.data[i] = fill_value;
}

__global__ internal void mat_rand_uniform_kernel(mat<f32> result, f32 min, f32 max, u64 seed)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= result.rows || col >= result.cols) return;
    size i = row*result.cols + col;

    curandState state = {};
    curand_init(seed, i, 0, &state);
    f32 val = curand_uniform(&state);
    result.data[i] = min + val * (max - min);
}

__global__ internal void mat_rand_gauss_kernel(mat<f32> result, f32 mean, f32 std_dev, u64 seed)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= result.rows || col >= result.cols) return;
    size i = row*result.cols + col;

    curandState state = {};
    curand_init(seed, i, 0, &state);
    f32 val = curand_normal(&state);
    result.data[i] = val*std_dev + mean;
}

__global__ internal void mat_rand_gauss_standard_kernel(mat<f32> result, u64 seed)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= result.rows || col >= result.cols) return;
    size i = row*result.cols + col;

    curandState state = {};
    curand_init(seed, i, 0, &state);
    result.data[i] = curand_normal(&state);
}

template <typename T>
__global__ internal void mat_copy_kernel(mat<T> result, mat<T> m)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= result.rows || col >= result.cols) return;

    size i = row*result.cols + col;
    result.data[i] = m.data[i];
}

template <typename T>
__global__ internal void transpose_kernel(mat<T> m, mat<T> result)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    result.data[col*result.cols + row] = m.data[row*m.cols + col];
}

template <typename T>
__global__ internal void mat_set_row_kernel(mat<T> m, vec<T> v, size row)
{
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (col >= m.cols) return;

    size i = row*m.cols + col;
    m.data[i] = v.data[col];
}

template <typename T>
__global__ internal void mat_set_col_kernel(mat<T> m, vec<T> v, size col)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    if (row >= m.rows) return;

    size i = row*m.cols + col;
    m.data[i] = v.data[row];
}

template <typename T>
__global__ internal void mat_set_row_range_kernel(mat<T> m, vec<T> v, size row, size row_offset)
{
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (col >= v.elements) return;

    size i = m.cols*row + (row_offset + col);
    m.data[i] = v.data[col];
}

template <typename T>
__global__ internal void mat_set_col_range_kernel(mat<T> m, vec<T> v, size col, size col_offset)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    if (row >= v.elements) return;

    size i = m.cols*(col_offset + row) + col;
    m.data[i] = v.data[row];
}

template <typename T>
__global__ internal void mat_add_kernel(mat<T> a, mat<T> b)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= a.rows || col >= b.cols) return;

    size i = a.cols*row + col;
    a.data[i] += b.data[i];
}

template <typename T>
__global__ internal void mat_add_vec_rows_kernel(mat<T> m, vec<T> v)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    m.data[i] += v.data[col];
}

template <typename T>
__global__ internal void mat_add_vec_cols_kernel(mat<T> m, vec<T> v)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    m.data[i] += v.data[row];
}

template <typename T>
__global__ internal void mat_scale_rows_kernel(mat<T> m, vec<T> scale)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    m.data[i] *= scale.data[row];
}

template <typename T>
__global__ internal void mat_scale_cols_kernel(mat<T> m, vec<T> scale)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    m.data[i] *= scale.data[col];
}

template <typename T>
__global__ internal void mat_scale_kernel(mat<T> m, T value)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    m.data[i] *= value;
}

template <typename T>
__global__ internal void mat_sum_rows_kernel(mat<T> m, vec<T> result)
{
    __shared__ T shared_mem[256];
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    int tid = threadIdx.x;
    shared_mem[tid] = m.data[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            shared_mem[tid] += shared_mem[tid + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        result.data[row] = shared_mem[0];
}

template <typename T>
__global__ internal void mat_sum_cols_kernel(mat<T> m, vec<T> result)
{
    __shared__ T shared_mem[256];
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= m.rows || col >= m.cols) return;

    size i = m.cols*row + col;
    int tid = threadIdx.x;
    T sum = 0;
    shared_mem[tid] = m.data[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
            shared_mem[tid] += shared_mem[tid + s];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        result.data[col] = shared_mem[0];
}

template <typename T>
__global__ internal void mat_mul_kernel(mat<T> a, mat<T> b, mat<T> result)
{
    // NOTE(lucas): rows and cols refer to result dimensions
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= a.rows || col >= b.cols) return;

    T sum = 0;
    for (size k = 0; k < a.cols; ++k)
        sum += a.data[a.cols*row + k]*b.data[b.cols*k + col];

    result.data[result.cols*row + col] = sum;
}

template <typename T>
__global__ internal void mat_had_kernel(mat<T> a, mat<T> b)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;
    if (row >= a.rows || col >= b.cols) return;

    size i = a.cols*row + col;
    a.data[i] *= b.data[i];
}

template <typename T>
void mat_to(mat<T>* m, Device device)
{
    mat<T> result = {};
    result.rows = m->rows;
    result.cols = m->cols;
    switch (m->device)
    {
        case Device_CPU:
        {
            if (device == Device_GPU)
            {
                size mem = m->rows*m->cols*sizeof(T);
                cuda_call(cudaMalloc(&result.data, mem));
                cuda_call(cudaMemcpy(result.data, m->data, mem, cudaMemcpyHostToDevice));
                result.device = Device_GPU;
                free(m->data);
                *m = result;
            }
        } break;

        case Device_GPU:
        {
            if (device == Device_CPU)
            {
                size mem = m->rows*m->cols*sizeof(T);
                result.data = (T*)malloc(mem);
                cuda_call(cudaMemcpy(result.data, m->data, mem, cudaMemcpyDeviceToHost));
                result.device = Device_CPU;
                cuda_call(cudaFree(m->data));
                *m = result;
            }
        } break;

        default: log_invalid_device(device); break;
    }
}

template <typename T>
void mat_free_data_gpu(mat<T> m)
{
    cuda_call(cudaFree(m.data));
}

template <typename T>
mat<T> mat_zeros_gpu(size rows, size cols)
{
    mat<T> result = {};
    result.rows = rows;
    result.cols = cols;
    result.device = Device_GPU;

    size mem = rows*cols*sizeof(T);
    cuda_call(cudaMalloc(&result.data, mem));
    cuda_call(cudaMemset(result.data, 0, mem));

    ASSERT(result.data, "GPU matrix allocation failed.");

    return result;
}

template <typename T>
mat<T> mat_full_gpu(size rows, size cols, T fill_value)
{
    mat<T> result = mat_zeros_gpu<T>(rows, cols);
    ThreadLayout layout = calc_thread_dim(rows, cols);
    mat_full_kernel<<<layout.grid_dim, layout.block_dim>>>(result, fill_value);
    cuda_call(cudaGetLastError());

    return result;
}

mat<f32> mat_rand_uniform_gpu(size rows, size cols, f32 min, f32 max)
{
    mat<f32> result = mat_zeros_gpu<f32>(rows, cols);
    ThreadLayout layout = calc_thread_dim(rows, cols);
    mat_rand_uniform_kernel<<<layout.grid_dim, layout.block_dim>>>(result, min, max, gpu_rng_global.seed);
    cuda_call(cudaGetLastError());

    return result;
}

mat<f32> mat_rand_gauss_gpu(size rows, size cols, f32 mean, f32 std_dev)
{
    mat<f32> result = mat_zeros_gpu<f32>(rows, cols);
    ThreadLayout layout = calc_thread_dim(rows, cols);
    mat_rand_gauss_kernel<<<layout.grid_dim, layout.block_dim>>>(result, mean, std_dev, gpu_rng_global.seed);
    cuda_call(cudaGetLastError());

    return result;
}

mat<f32> mat_rand_gauss_standard_gpu(size rows, size cols)
{
    mat<f32> result = mat_zeros_gpu<f32>(rows, cols);
    ThreadLayout layout = calc_thread_dim(rows, cols);
    mat_rand_gauss_standard_kernel<<<layout.grid_dim, layout.block_dim>>>(result, gpu_rng_global.seed);
    cuda_call(cudaGetLastError());

    return result;
}

template <typename T>
mat<T> mat_copy_gpu(mat<T> m)
{
    mat<T> result = mat_zeros_gpu<T>(m.rows, m.cols);
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_copy_kernel<<<layout.grid_dim, layout.block_dim>>>(result, m);
    cuda_call(cudaGetLastError());

    return result;
}

template <typename T>
mat<T> transpose_gpu(mat<T> m)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat<T> result = mat_zeros_gpu<T>(m.cols, m.rows);
    transpose_kernel<<<layout.grid_dim, layout.block_dim>>>(m, result);
    cuda_call(cudaGetLastError());

    m.data = result.data;
    m.rows = result.rows;
    m.cols = result.cols;
    return m;
}

template <typename T>
void mat_set_row_gpu(mat<T> m, vec<T> v, size row)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_set_row_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v, row);
    cuda_call(cudaGetLastError());
}

template <typename T>
void mat_set_col_gpu(mat<T> m, vec<T> v, size col)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_set_col_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v, col);
    cuda_call(cudaGetLastError());
}

template <typename T>
void mat_set_row_range_gpu(mat<T> m, vec<T> v, size row, size row_offset)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_set_row_range_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v, row, row_offset);
    cuda_call(cudaGetLastError());
}

template <typename T>
void mat_set_col_range_gpu(mat<T> m, vec<T> v, size col, size col_offset)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_set_col_range_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v, col, col_offset);
    cuda_call(cudaGetLastError());
}

template <typename T>
void mat_add_gpu(mat<T> a, mat<T> b)
{
    ThreadLayout layout = calc_thread_dim(a.rows, a.cols);
    mat_add_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b);
    cuda_call(cudaGetLastError());
}

template <typename T>
void mat_add_vec_gpu(mat<T> m, vec<T> v, Axis axis)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);

    switch (axis)
    {
        case Axis_Rows:
        {
            ASSERTF(v.elements == m.cols,
                    "For a row-wise add, the vector must have the same number of elements as the matrix has columns "
                    "(matrix columns: %llu, vector elements: %llu).", m.cols, v.elements);

            mat_add_vec_rows_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v);
            cuda_call(cudaGetLastError());
        } break;

        case Axis_Cols:
        {
            ASSERTF(v.elements == m.rows,
                    "For a column-wise add, the vector must have the same number of elements as the matrix has rows "
                    "(matrix rows: %llu, vector elements: %llu).", m.rows, v.elements);

            mat_add_vec_cols_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v);
            cuda_call(cudaGetLastError());
        } break;

        default: log_invalid_axis(axis); break;
    }
}

template <typename T>
void mat_scale_gpu(mat<T> m, T value)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_scale_kernel<<<layout.grid_dim, layout.block_dim>>>(m, value);
    cuda_call(cudaGetLastError());
}

template <typename T>
void mat_scale_gpu(mat<T> m, vec<T> scale, Axis axis)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    switch (axis)
    {
        case Axis_Rows:
        {
            ASSERTF(scale.elements == m.rows,
                    "For a row-wise scale, the vector must have the same number of elements as the matrix has rows "
                    "(matrix rows: %llu, vector elements: %llu).", m.rows, scale.elements);

            mat_scale_rows_kernel<<<layout.grid_dim, layout.block_dim>>>(m, scale);
            cuda_call(cudaGetLastError());
        } break;

        case Axis_Cols:
        {
            ASSERTF(scale.elements == m.cols,
                    "For a column-wise scale, the vector must have the same number of elements as the matrix has cols "
                    "(matrix columns: %llu, vector elements: %llu).", m.cols, scale.elements);

            mat_scale_cols_kernel<<<layout.grid_dim, layout.block_dim>>>(m, scale);
            cuda_call(cudaGetLastError());
        } break;

        default: log_invalid_axis(axis); break;
    }
}

template <typename T>
mat<T> mat_mul_gpu(mat<T> a, mat<T> b)
{
    mat<T> result = mat_zeros_gpu<T>(a.rows, b.cols);
    ThreadLayout layout = calc_thread_dim(a.rows, b.cols);
    mat_mul_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b, result);
    cuda_call(cudaGetLastError());

    return result;
}

template <typename T>
void mat_had_gpu(mat<T> a, mat<T> b)
{
    ThreadLayout layout = calc_thread_dim(a.rows, a.cols);
    mat_had_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b);
    cuda_call(cudaGetLastError());
}

template <typename T>
vec<T> mat_sum_gpu(mat<T> m, Axis axis)
{
    vec<T> result = {};
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    size shared_mem_size = layout.block_dim.x * sizeof(T);
    switch (axis)
    {
        case Axis_Rows:
        {
            result = vec_zeros_gpu<T>(m.rows);
            mat_sum_rows_kernel<<<layout.grid_dim, layout.block_dim, shared_mem_size>>>(m, result);
            cuda_call(cudaGetLastError());
        } break;

        case Axis_Cols:
        {
            result = vec_zeros_gpu<T>(m.cols);
            mat_sum_cols_kernel<<<layout.grid_dim, layout.block_dim, shared_mem_size>>>(m, result);
            cuda_call(cudaGetLastError());
        } break;

        default: log_invalid_axis(axis); break;
    }

    return result;
}

#define MAT_TO(T) INST_TEMPLATE(mat_to, void, T, (mat<T>* m, Device device))
#define MAT_FREE_DATA_GPU(T) INST_TEMPLATE(mat_free_data_gpu, void, T, (mat<T> m))
#define MAT_ZEROS_GPU(T) INST_TEMPLATE(mat_zeros_gpu, mat<T>, T, (size rows, size cols))
#define MAT_FULL_GPU(T) INST_TEMPLATE(mat_full_gpu, mat<T>, T, (size rows, size cols, T fill_value))
#define MAT_COPY_GPU(T) INST_TEMPLATE(mat_copy_gpu, mat<T>, T, (mat<T> m))
#define TRANSPOSE_GPU(T) INST_TEMPLATE(transpose_gpu, mat<T>, T, (mat<T> m))
#define MAT_SET_ROW_GPU(T) INST_TEMPLATE(mat_set_row_gpu, void, T, (mat<T> m, vec<T> data, size row))
#define MAT_SET_COL_GPU(T) INST_TEMPLATE(mat_set_col_gpu, void, T, (mat<T> m, vec<T> data, size col))
#define MAT_SET_ROW_RANGE_GPU(T) INST_TEMPLATE(mat_set_row_range_gpu, void, T, (mat<T> m, vec<T> data, size row, size row_offset))
#define MAT_SET_COL_RANGE_GPU(T) INST_TEMPLATE(mat_set_col_range_gpu, void, T, (mat<T> m, vec<T> data, size col, size col_offset))
#define MAT_ADD_GPU(T) INST_TEMPLATE(mat_add_gpu, void, T, (mat<T> a, mat<T> b))
#define MAT_ADD_VEC_GPU(T) INST_TEMPLATE(mat_add_vec_gpu, void, T, (mat<T> m, vec<T> v, Axis axis))
#define MAT_SCALE_GPU(T) INST_TEMPLATE(mat_scale_gpu, void, T, (mat<T> m, T scale))
#define MAT_SCALE_GPU_VEC(T) INST_TEMPLATE(mat_scale_gpu, void, T, (mat<T> m, vec<T> scale, Axis axis))
#define MAT_MUL_GPU(T) INST_TEMPLATE(mat_mul_gpu, mat<T>, T, (mat<T> a, mat<T> b))
#define MAT_HAD_GPU(T) INST_TEMPLATE(mat_had_gpu, void, T, (mat<T> a, mat<T> b))
#define MAT_SUM_GPU(T) INST_TEMPLATE(mat_sum_gpu, vec<T>, T, (mat<T> m, Axis axis))

INST_ALL_TYPES(MAT_TO)
INST_ALL_TYPES(MAT_FREE_DATA_GPU)
INST_ALL_TYPES(MAT_ZEROS_GPU)
INST_ALL_TYPES(MAT_FULL_GPU)
INST_ALL_TYPES(MAT_COPY_GPU)
INST_ALL_TYPES(TRANSPOSE_GPU)
INST_ALL_TYPES(MAT_SET_ROW_GPU)
INST_ALL_TYPES(MAT_SET_COL_GPU)
INST_ALL_TYPES(MAT_SET_ROW_RANGE_GPU)
INST_ALL_TYPES(MAT_SET_COL_RANGE_GPU)
INST_ALL_TYPES(MAT_ADD_GPU)
INST_ALL_TYPES(MAT_ADD_VEC_GPU)
INST_ALL_TYPES(MAT_SCALE_GPU)
INST_ALL_TYPES(MAT_SCALE_GPU_VEC)
INST_ALL_TYPES(MAT_MUL_GPU)
INST_ALL_TYPES(MAT_HAD_GPU)
INST_ALL_TYPES(MAT_SUM_GPU)

#undef MAT_TO
#undef MAT_FREE_DATA_GPU
#undef MAT_ZEROS_GPU
#undef MAT_FULL_GPU
#undef MAT_COPY_GPU
#undef TRANSPOSE_GPU
#undef MAT_SET_ROW_GPU
#undef MAT_SET_COL_GPU
#undef MAT_SET_ROW_RANGE_GPU
#undef MAT_SET_COL_RANGE_GPU
#undef MAT_ADD_GPU
#undef MAT_ADD_VEC_GPU
#undef MAT_SCALE_GPU
#undef MAT_SCALE_GPU_VEC
#undef MAT_HAD_GPU
#undef MAT_SUM_GPU
