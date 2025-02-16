#include "matrix.h"
#include "cuda/matrix_cuda.h"
#include "rng.cpp"

template <typename T>
internal mat<T> mat_init(size rows, size cols, T* data, Device device)
{
    mat<T> result = {};
    result.rows = rows;
    result.cols = cols;
    result.data = data;

    // TODO(lucas): Verify that the device and pointer location match?
    result.device = device;

    return result;
}

template <typename T>
internal mat<T> mat_zeros(size rows, size cols, Device device)
{
    mat<T> result = {};

    switch (device)
    {
        case Device_CPU:
        {
            result.rows = rows;
            result.cols = cols;
            result.data = (T*)calloc(rows*cols, sizeof(T));
            ASSERT(result.data, "CPU matrix allocation failed.");
        } break;

        case Device_GPU: result = mat_zeros_gpu<T>(rows, cols); break;

        default: log_invalid_device(device); break;
    }

    return result;
}

template <typename T>
internal mat<T> mat_full(size rows, size cols, T fill_value, Device device)
{
    mat<T> result = {};
    switch (device)
    {
        case Device_CPU:
        {
            result = mat_zeros<T>(rows, cols);
            for (size row = 0; row < result.rows; ++row)
            {
                for (size col = 0; col < result.cols; ++col)
                    result(row, col) = fill_value;
            }
        } break;

        case Device_GPU: result = mat_full_gpu<T>(rows, cols, fill_value); break;

        default: log_invalid_device(device); break;
    }

    return result;
}

template <typename T>
internal mat<T> mat_copy(mat<T> m)
{
    mat<T> result = {};
    
    switch (m.device)
    {
        case Device_CPU:
        {
            result = mat_zeros<T>(m.rows, m.cols);
            for (size row = 0; row < m.rows; ++row)
            {
                for (size col = 0; col < m.cols; ++col)
                    result(row, col) = m(row, col);
            }
        } break;

        case Device_GPU: result = mat_copy_gpu(m); break;

        default: log_invalid_device(m.device); break;
    }
    
    return result;
}

internal mat<f32> mat_rand_uniform(size rows, size cols, f32 min, f32 max, Device device)
{
    mat<f32> result ={};

    switch (device)
    {
        case Device_CPU:
        {
            result = mat_zeros<f32>(rows, cols);
            for (size row = 0; row < rows; ++row)
            {
                for (size col = 0; col < cols; ++col)
                   result(row, col) = rand_f32_uniform(min, max);
            }
        } break;

        case Device_GPU: result = mat_rand_uniform_gpu(rows, cols, min, max); break;

        default: break;
    }
    
    return result;
}

internal mat<f32> mat_rand_gauss(size rows, size cols, f32 mean, f32 std_dev, Device device)
{
    mat<f32> result = {};
    
    switch (device)
    {
        case Device_CPU:
        {
            result = mat_zeros<f32>(rows, cols);
            for (size row = 0; row < rows; ++row)
            {
                for (size col = 0; col < cols; ++col)
                   result(row, col) = rand_f32_gauss(mean, std_dev);
            }
            
        } break;

        case Device_GPU: result = mat_rand_gauss_gpu(rows, cols, mean, std_dev); break;

        default: break;
    }

    return result;
}

internal mat<f32> mat_rand_gauss_standard(size rows, size cols, Device device)
{
    mat<f32> result = {};
    
    switch (device)
    {
        case Device_CPU:
        {
            result = mat_zeros<f32>(rows, cols);
            for (size row = 0; row < rows; ++row)
            {
                for (size col = 0; col < cols; ++col)
                   result(row, col) = rand_f32_gauss_standard();
            }
        } break;

        case Device_GPU: result = mat_rand_gauss_standard_gpu(rows, cols); break;

        default: break;
    }

    return result;
}

template <typename T>
internal void mat_set_row(mat<T> m, vec<T> data, size row)
{
    // To set a row, the vector must have the same number of elements as
    // the matrix has columns.
    ASSERT(m.cols == data.elements, "Mismatch in matrix columns and vector elements.\n");

    switch (m.device)
    {
        case Device_CPU:
        {
            for (size col = 0; col < m.cols; ++col)
                m(row, col) = data[col];
        } break;

        case Device_GPU: mat_set_row_gpu(m, data, row); break;

        default: log_invalid_device(m.device); break;
    }
}

template <typename T>
internal void mat_set_col(mat<T> m, vec<T> data, size col)
{
    // To set a column, the vector must have the same number of elements as
    // the matrix has rows.
    ASSERT(m.rows == data.elements, "Mismatch in matrix rows and vector elements.\n");

    switch (m.device)
    {
        case Device_CPU:
        {
            for (size row = 0; row < m.rows; ++col)
                m(row, col) = data[row];
        } break;

        case Device_GPU: mat_set_col_gpu(m, data, col); break;

        default: log_invalid_device(m.device); break;
    }
}

template <typename T>
vec<T> mat_get_row(mat<T> m, size row)
{
    ASSERT(row < m.rows, "Row out of bounds.\n");

    vec<T> result = {};
    result.elements = m.cols;
    result.device = m.device;
    result.data = m.data + row*m.cols;

    return result;
}

template <typename T>
vec<T> mat_get_col(mat<T> m, size col)
{
    ASSERT(col < m.cols, "Column out of bounds.\n");

    vec<T> result = {};
    result.elements = m.rows;
    result.device = m.device;
    result.data = m.data + col;

    return result;
}

template <typename T>
internal void mat_set_row_range(mat<T> m, vec<T> data, size row, size row_offset)
{
    ASSERT(m.cols >= data.elements + row_offset, "Not enough columns in matrix after offset to accommodate vector.\n");

    switch (m.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < data.elements; ++i)
                m(row, row_offset + i) =  data[i];
        } break;

        case Device_GPU: mat_set_row_range_gpu(m, data, row, row_offset); break;

        default: log_invalid_device(m.device); break;
    }
}

template <typename T>
internal void mat_set_col_range(mat<T> m, vec<T> data, size col, size col_offset)
{
    ASSERT(m.rows >= data.elements + col_offset, "Not enough rows in matrix after offset to accommodate vector.\n");

    switch (m.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < data.elements; ++i)
                m(col_offset + i, col) = data[i];
        } break;

        case Device_GPU: mat_set_col_range_gpu(m, data, col, col_offset); break;

        default: log_invalid_device(m.device); break;
    }
}

// TODO(lucas): Make sure numbers are properly aligned, taking into account things like negative signs.
template <typename T>
internal void mat_print(mat<T> m)
{
    b32 was_on_gpu = m.device == Device_GPU;
    if (was_on_gpu)
        mat_to(&m, Device_CPU);

    // Compute max width needed for printing
    int width = 0;
    for (size row = 0; row < m.rows; ++row)
    {
        for (size col = 0; col < m.cols; ++col)
        {
            int w = 0;
            if constexpr      (std::is_same_v<T, u8>)  w = snprintf(0, 0, "%hhu", m(row, col));
            else if constexpr (std::is_same_v<T, u16>) w = snprintf(0, 0, "%hu",  m(row, col));
            else if constexpr (std::is_same_v<T, u32>) w = snprintf(0, 0, "%u",   m(row, col));
            else if constexpr (std::is_same_v<T, u64>) w = snprintf(0, 0, "%llu", m(row, col));
            else if constexpr (std::is_same_v<T, i8>)  w = snprintf(0, 0, "%hhd", m(row, col));
            else if constexpr (std::is_same_v<T, i16>) w = snprintf(0, 0, "%hd",  m(row, col));
            else if constexpr (std::is_same_v<T, i32>) w = snprintf(0, 0, "%d",   m(row, col));
            else if constexpr (std::is_same_v<T, i64>) w = snprintf(0, 0, "%lld", m(row, col));
            else if constexpr (std::is_same_v<T, f32>) w = snprintf(0, 0, "%f",   m(row, col));
            else if constexpr (std::is_same_v<T, f64>) w = snprintf(0, 0, "%f",   m(row, col));
            
            if (width < w)
                width = w;
        }
    }

    // Print
    printf("[");
    for (size row = 0; row < m.rows; ++row)
    {
        if (row > 0) printf(" ");
        printf("[");
        for (size col = 0; col < m.cols; ++col)
        {
            if (col != 0) printf(", ");

            if constexpr (std::is_same_v<T, u8>)       printf("%hhu", m(row, col));
            else if constexpr (std::is_same_v<T, u16>) printf("%hu",  m(row, col));
            else if constexpr (std::is_same_v<T, u32>) printf("%u",   m(row, col));
            else if constexpr (std::is_same_v<T, u64>) printf("%llu", m(row, col));
            else if constexpr (std::is_same_v<T, i8>)  printf("%hhd", m(row, col));
            else if constexpr (std::is_same_v<T, i16>) printf("%hd",  m(row, col));
            else if constexpr (std::is_same_v<T, i32>) printf("%d",   m(row, col));
            else if constexpr (std::is_same_v<T, i64>) printf("%lld", m(row, col));
            else if constexpr (std::is_same_v<T, f32>) printf("%f",   m(row, col));
            else if constexpr (std::is_same_v<T, f64>) printf("%f",   m(row, col));
        }
        printf("]");
        
        if(row < m.rows - 1)
            printf(",\n");
    }
    printf("]\n");

    if (was_on_gpu)
    mat_to(&m, Device_GPU);
}

template <typename T>
internal mat<T> mat_add(mat<T> a, mat<T> b, b32 in_place)
{
    ASSERT(a.rows == b.rows, "Matrix addition requires the matrices to be the same size.\n");
    ASSERT(a.cols == b.cols, "Matrix addition requires the matrices to be the same size.\n");
    ASSERT(a.device == b.device, "The matrices must be on the same device.\n");

    mat<T> result = in_place ? a : mat_copy(a);

    switch (result.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.rows; ++i)
            {
                for (size j = 0; j < result.cols; ++j)
                    result(i, j) += b(i, j);
            }
        } break;

        case Device_GPU: mat_add_gpu(result, b); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

template <typename T>
internal void mat_add_vec_cpu(mat<T> m, vec<T> v, Axis axis)
{
    switch (axis)
    {
        case Axis_Rows:
        {
            ASSERT(v.elements == m.cols,
                   "For a row-wise add, the vector must have the same number of elements as the matrix has columns.");

            for (size row = 0; row < m.rows; ++row)
            {
                for (size col = 0; col < m.cols; ++col)
                    m(row, col) += v[col];
            }
        } break;

        case Axis_Cols:
        {
            ASSERT(v.elements == m.rows,
                   "For a column-wise add, the vector must have the same number of elements as the matrix has rows.");
            
            for (size row = 0; row < m.rows; ++row)
            {
                for (size col = 0; col < m.cols; ++col)
                    m(row, col) += v[row];
            }
        } break;

        default: log_invalid_axis(axis); break;
    }
}

template <typename T>
internal mat<T> mat_add_vec(mat<T> m, vec<T> v, Axis axis, b32 in_place)
{
    ASSERT(m.device == v.device, "The matrix and vector must be on the same device.\n");

    mat<T> result = in_place ? m : mat_copy(m);
    switch (m.device)
    {
        case Device_CPU: mat_add_vec_cpu(m, v, axis); break;
        case Device_GPU: mat_add_vec_gpu(m, v, axis); break;
        default: log_invalid_device(result.device); break;
    }

    return result;
}

template <typename T>
internal void mat_free_data(mat<T> a)
{
    switch (a.device)
    {
        case Device_CPU: free(a.data); break;
        case Device_GPU: mat_free_data_gpu(a); break;
        default: log_invalid_device(a.device); break;
    }
}

template <typename T>
internal mat<T> mat_scale(mat<T> m, T scale, b32 in_place)
{
    mat<T> result = in_place ? m : mat_copy(m);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size row = 0; row < result.rows; ++row)
            {
                for (size col = 0; col < result.cols; ++col)
                    result(row, col) *= scale;
            }
        } break;

        case Device_GPU: mat_scale_gpu(result, scale); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

template <typename T>
internal void mat_scale_cpu(mat<T> m, vec<T> scale, Axis axis)
{
    switch (axis)
    {
        case Axis_Rows:
        {
            ASSERT(scale.elements == m.cols,
                "For a row-wise scale, the vector must have the same number of elements as the matrix has columns.");

            for (size row = 0; row < m.rows; ++row)
            {
                for (size col = 0; col < m.cols; ++col)
                    m(row, col) *= scale[col];
            }
        } break;

        case Axis_Cols:
        {
            ASSERT(scale.elements == m.cols,
                "For a column-wise scale, the vector must have the same number of elements as the matrix has rows.");

            for (size row = 0; row < m.rows; ++row)
            {
                for (size col = 0; col < m.cols; ++col)
                    m(row, col) *= scale[row];
            }
        } break;

        default: log_invalid_axis(axis); break;
    }
}

template <typename T>
internal mat<T> mat_scale(mat<T> m, vec<T> scale, Axis axis, b32 in_place)
{
    ASSERT(m.device == scale.device, "The matrix and vector must be on the same device.\n");

    mat<T> result = in_place ? m : mat_copy(m);
    switch (result.device)
    {
        case Device_CPU: mat_scale_cpu(result, scale, axis); break;
        case Device_GPU: mat_scale_gpu(result, scale, axis); break;
        default: log_invalid_device(result.device); break;
    }

    return result;
}

template <typename T>
internal mat<T> mat_mul(mat<T> a, mat<T> b)
{
    ASSERT(a.cols == b.rows, "For A*B to be valid, the number of columns in A must equal the number of rows in B.\n");
    ASSERT(a.device == b.device, "The matrices must be on the same device.\n");

    mat<T> result = {};
    switch (a.device)
    {
        case Device_CPU:
        {
            // Shape of output matrix is determined by the number of rows in A and the number of columns in B
            result = mat_zeros<T>(a.rows, b.cols);
            for (size i = 0; i < result.rows; ++i)
            {
                for (size j = 0; j < result.cols; ++j)
                {
                    T sum = 0;
                    for (size k = 0; k < a.cols; ++k)
                        sum += a(i, k) * b(k, j);
                    
                    result(i, j) = sum;
                }
            }
        } break;

        case Device_GPU: result = mat_mul_gpu(a, b); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

template <typename T>
internal mat<T> mat_had(mat<T> a, mat<T> b, b32 in_place)
{
    ASSERT(a.rows == b.rows, "The Hadamard product requires the matrices to be the same size.\n");
    ASSERT(a.cols == b.cols, "The Hadamard product requires the matrices to be the same size.\n");
    ASSERT(a.device == b.device, "The matrices must be on the same device.\n");

    mat<T> result = in_place ? a : mat_copy(a);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size row = 0; row < result.rows; ++row)
            {
                for (size col = 0; col < result.cols; ++col)
                    result(row, col) *= b(row, col);
            }
        } break;

        case Device_GPU: mat_had_gpu(result, b); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

template <typename T>
internal vec<T> mat_sum_cpu(mat<T> m, Axis axis)
{
    vec<T> result = {};
    switch (axis)
    {
        case Axis_Rows:
        {
            result = vec_zeros<T>(m.rows);
            for (size col = 0; col < m.cols; ++col)
            {
                for (size row = 0; row < m.rows; ++row)
                    result[row] += m(row, col);
            }
        } break;

        case Axis_Cols:
        {
            result = vec_zeros<T>(m.cols);
            for (size col = 0; col < m.cols; ++col)
            {
                for (size row = 0; row < m.rows; ++row)
                    result[col] += m(row, col);
            }
        } break;

        default: log_invalid_axis(axis); break;
    }

    return result;
}

template <typename T>
internal vec<T> mat_sum(mat<T> m, Axis axis)
{
    vec<T> result = {};
    switch (m.device)
    {
        case Device_CPU: result = mat_sum_cpu(m, axis); break;
        case Device_GPU: result = mat_sum_gpu(m, axis); break;
        default: log_invalid_device(m.device); break;
    }

    return result;
}
