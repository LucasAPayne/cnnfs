#include "cuda_base.h"
#include "matrix_cuda.h"
#include "vector_cuda.h"

// TODO(lucas): Temporary
template <typename T>
internal vec<T> mat_get_row(mat<T> m, size row)
{
    ASSERT(row < m.rows);

    vec<T> result = {};
    result.elements = m.cols;
    result.device = m.device;
    result.data = m.data + row*m.cols;

    return result;
}

template <typename T>
internal vec<T> mat_get_col(mat<T> m, size col)
{
    ASSERT(col < m.cols);

    vec<T> result = {};
    result.elements = m.rows;
    result.device = m.device;
    result.data = m.data + col;

    return result;
}

template <typename T>
__global__ internal void mat_full_kernel(mat<T> result, T fill_value)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= result.rows || col >= result.cols)
        return;

    size i = row*result.cols + col;
    result.data[i] = fill_value;
}

template <typename T>
__global__ internal void mat_set_row_kernel(mat<T> m, vec<T> v, size row)
{
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (col >= m.cols)
        return;

    size i = row*m.cols + col;
    m.data[i] = v.data[col];
}

template <typename T>
__global__ internal void mat_set_col_kernel(mat<T> m, vec<T> v, size col)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;

    if (row >= m.rows)
        return;

    size i = row*m.cols + col;
    m.data[i] = v.data[row];
}

template <typename T>
__global__ internal void mat_set_row_range_kernel(mat<T> m, vec<T> v, size row, size row_offset)
{
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (col >= v.elements)
        return;

    size i = m.cols*row + (row_offset + col);
    m.data[i] = v.data[col];
}

template <typename T>
__global__ internal void mat_set_col_range_kernel(mat<T> m, vec<T> v, size col, size col_offset)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;

    if (row >= v.elements)
        return;
    
    size i = m.cols*(col_offset + row) + col;
    m.data[i] = v.data[row];
}

template <typename T>
__global__ internal void mat_stretch_cols_kernel(mat<T> orig, mat<T> target, mat<T> result)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= target.rows || col >= target.cols)
        return;

    result.data[result.cols*row + col] = orig.data[col];
}

template <typename T>
__global__ internal void mat_stretch_rows_kernel(mat<T> orig, mat<T> target, mat<T> result)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= target.rows || col >= target.cols)
        return;

    result.data[result.cols*row + col] = orig.data[row];
}

template <typename T>
__global__ internal void mat_add_kernel(mat<T> a, mat<T> b)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= a.rows || col >= b.cols)
        return;
    
    size i = a.cols*row + col;
    a.data[i] += b.data[i];
}

template <typename T>
__global__ internal void mat_scale_kernel(mat<T> m, T value)
{
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= m.rows || col >= m.cols)
        return;
    
    size i = m.cols*row + col;
    m.data[i] *= value;
}

template <typename T>
__global__ internal void mat_mul_kernel(mat<T> a, mat<T> b, mat<T> result)
{
    // NOTE(lucas): rows and cols refer to result dimensions
    int row = threadIdx.y + blockIdx.y*blockDim.y;
    int col = threadIdx.x + blockIdx.x*blockDim.x;

    if (row >= a.rows || col >= b.cols)
        return;

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

    if (row >= a.rows || col >= b.cols)
        return;
    
    size i = a.cols*row + col;
    a.data[i] *= b.data[i];
}

template <typename T>
void mat_to(mat<T>* m, Device device)
{
    mat<T> result = {};
    result.rows = m->rows;
    result.cols = m->cols;
    switch(m->device)
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

        default: break;
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

    return result;
}

template <typename T>
mat<T> mat_full_gpu(size rows, size cols, T fill_value)
{
    mat<T> result = mat_zeros_gpu<T>(rows, cols);
    ThreadLayout layout = calc_thread_dim(rows, cols);
    mat_full_kernel<<<layout.grid_dim, layout.block_dim>>>(result, fill_value);
    sync_kernel();

    return result;
}

template <typename T>
void mat_set_row_gpu(mat<T> m, vec<T> v, size row)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_set_row_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v, row);
    sync_kernel();
}

template <typename T>
void mat_set_col_gpu(mat<T> m, vec<T> v, size col)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_set_col_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v, col);
    sync_kernel();
}

template <typename T>
void mat_set_row_range_gpu(mat<T> m, vec<T> v, size row, size row_offset)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_set_row_range_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v, row, row_offset);
    sync_kernel();
}

template <typename T>
void mat_set_col_range_gpu(mat<T> m, vec<T> v, size col, size col_offset)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_set_col_range_kernel<<<layout.grid_dim, layout.block_dim>>>(m, v, col, col_offset);
    sync_kernel();
}

template <typename T>
internal mat<T> mat_stretch_rows_gpu(mat<T> orig, mat<T> target)
{
    mat<T> result = mat_zeros_gpu<T>(target.rows, target.cols);
    ThreadLayout layout = calc_thread_dim(target.rows, target.cols);
    mat_stretch_cols_kernel<<<layout.grid_dim, layout.block_dim>>>(orig, target, result);
    sync_kernel();

    return result;
}

template <typename T>
internal mat<T> mat_stretch_cols_gpu(mat<T> orig, mat<T> target)
{
    mat<T> result = mat_zeros_gpu<T>(target.rows, target.cols);
    ThreadLayout layout = calc_thread_dim(target.rows, target.cols);
    mat_stretch_rows_kernel<<<layout.grid_dim, layout.block_dim>>>(orig, target, result);
    sync_kernel();

    return result;
}

template <typename T>
void mat_add_gpu(mat<T> a, mat<T> b)
{
    ThreadLayout layout = calc_thread_dim(a.rows, a.cols);
    mat_add_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b);
    sync_kernel();
}

template <typename T>
void mat_scale_gpu(mat<T> m, T value)
{
    ThreadLayout layout = calc_thread_dim(m.rows, m.cols);
    mat_scale_kernel<<<layout.grid_dim, layout.block_dim>>>(m, value);
    sync_kernel();
}

template <typename T>
void mat_scale_gpu(mat<T> m, vec<T> scale, Axis axis)
{
    switch (axis)
    {
        case Axis_Rows:
        {
            for (size i = 0; i < m.rows; ++i)
            {
                vec<T> row = mat_get_row(m, i);
                T c;
                cuda_call(cudaMemcpy(&c, scale.data + i, sizeof(T), cudaMemcpyDeviceToHost));
                vec_scale_gpu(row, c);
            }
        } break;

        case Axis_Cols:
        {
            for (size i = 0; i < m.cols; ++i)
            {
                vec<T> col = mat_get_col(m, i);
                T c;
                cuda_call(cudaMemcpy(&c, scale.data + i, sizeof(T), cudaMemcpyDeviceToHost));
                vec_scale_gpu(col, c);
            }
        } break;

        default: break;
    }
}

template <typename T>
mat<T> mat_mul_gpu(mat<T> a, mat<T> b)
{
    mat<T> result = mat_zeros_gpu<T>(a.rows, b.cols);
    ThreadLayout layout = calc_thread_dim(a.rows, b.cols);
    mat_mul_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b, result);
    sync_kernel();

    return result;
}

template <typename T>
void mat_had_gpu(mat<T> a, mat<T> b)
{
    ThreadLayout layout = calc_thread_dim(a.rows, a.cols);
    mat_had_kernel<<<layout.grid_dim, layout.block_dim>>>(a, b);
    sync_kernel();
}

// TODO(lucas): Perform sums in parallel.
template <typename T>
vec<T> mat_sum_gpu(mat<T> m, Axis axis)
{
    vec<T> result = {};
    switch (axis)
    {
        case Axis_Rows:
        {
            result = vec_zeros_gpu<T>(m.rows);
            for (size i = 0; i < m.rows; ++i)
            {
                vec<T> row = mat_get_row(m, i);
                T sum = vec_sum_gpu(row);
                cuda_call(cudaMemcpy(result.data + i, &sum, sizeof(T), cudaMemcpyHostToDevice));
            }
        } break;

        case Axis_Cols:
        {
            result = vec_zeros_gpu<T>(m.cols);
            for (size i = 0; i < m.cols; ++i)
            {
                vec<T> col = mat_get_col(m, i);
                T sum = vec_sum_gpu(col);
                cuda_call(cudaMemcpy(result.data + i, &sum, sizeof(T), cudaMemcpyHostToDevice));
            }
        } break;

        default: break;
    }

    return result;
}

#define MAT_TO(T) INST_TEMPLATE(mat_to, void, T, (mat<T>* m, Device device))
#define MAT_FREE_DATA_GPU(T) INST_TEMPLATE(mat_free_data_gpu, void, T, (mat<T> m))
#define MAT_ZEROS_GPU(T) INST_TEMPLATE(mat_zeros_gpu, mat<T>, T, (size rows, size cols))
#define MAT_FULL_GPU(T) INST_TEMPLATE(mat_full_gpu, mat<T>, T, (size rows, size cols, T fill_value))
#define MAT_SET_ROW_GPU(T) INST_TEMPLATE(mat_set_row_gpu, void, T, (mat<T> m, vec<T> data, size row))
#define MAT_SET_COL_GPU(T) INST_TEMPLATE(mat_set_col_gpu, void, T, (mat<T> m, vec<T> data, size col))
#define MAT_SET_ROW_RANGE_GPU(T) INST_TEMPLATE(mat_set_row_range_gpu, void, T, (mat<T> m, vec<T> data, size row, size row_offset))
#define MAT_SET_COL_RANGE_GPU(T) INST_TEMPLATE(mat_set_col_range_gpu, void, T, (mat<T> m, vec<T> data, size col, size col_offset))
#define MAT_STRETCH_COLS_GPU(T) INST_TEMPLATE(mat_stretch_cols_gpu, mat<T>, T, (mat<T> orig, mat<T> target))
#define MAT_STRETCH_ROWS_GPU(T) INST_TEMPLATE(mat_stretch_rows_gpu, mat<T>, T, (mat<T> orig, mat<T> target))
#define MAT_ADD_GPU(T) INST_TEMPLATE(mat_add_gpu, void, T, (mat<T> a, mat<T> b))
#define MAT_SCALE_GPU(T) INST_TEMPLATE(mat_scale_gpu, void, T, (mat<T> m, T scale))
#define MAT_SCALE_GPU_VEC(T) INST_TEMPLATE(mat_scale_gpu, void, T, (mat<T> m, vec<T> scale, Axis axis))
#define MAT_MUL_GPU(T) INST_TEMPLATE(mat_mul_gpu, mat<T>, T, (mat<T> a, mat<T> b))
#define MAT_HAD_GPU(T) INST_TEMPLATE(mat_had_gpu, void, T, (mat<T> a, mat<T> b))
#define MAT_SUM_GPU(T) INST_TEMPLATE(mat_sum_gpu, vec<T>, T, (mat<T> m, Axis axis))

INST_ALL_TYPES(MAT_TO)
INST_ALL_TYPES(MAT_FREE_DATA_GPU)
INST_ALL_TYPES(MAT_ZEROS_GPU)
INST_ALL_TYPES(MAT_FULL_GPU)
INST_ALL_TYPES(MAT_SET_ROW_GPU)
INST_ALL_TYPES(MAT_SET_COL_GPU)
INST_ALL_TYPES(MAT_SET_ROW_RANGE_GPU)
INST_ALL_TYPES(MAT_SET_COL_RANGE_GPU)
INST_ALL_TYPES(MAT_STRETCH_COLS_GPU)
INST_ALL_TYPES(MAT_STRETCH_ROWS_GPU)
INST_ALL_TYPES(MAT_ADD_GPU)
INST_ALL_TYPES(MAT_SCALE_GPU)
INST_ALL_TYPES(MAT_SCALE_GPU_VEC)
INST_ALL_TYPES(MAT_MUL_GPU)
INST_ALL_TYPES(MAT_HAD_GPU)
INST_ALL_TYPES(MAT_SUM_GPU)

#undef MAT_TO
#undef MAT_FREE_DATA_GPU
#undef MAT_ZEROS_GPU
#undef MAT_FULL_GPU
#undef MAT_SET_ROW_GPU
#undef MAT_SET_COL_GPU
#undef MAT_SET_ROW_RANGE_GPU
#undef MAT_SET_COL_RANGE_GPU
#undef MAT_STRETCH_COLS_GPU
#undef MAT_STRETCH_ROWS_GPU
#undef MAT_ADD_GPU
#undef MAT_SCALE_GPU
#undef MAT_SCALE_GPU_VEC
#undef MAT_HAD_GPU
#undef MAT_SUM_GPU
