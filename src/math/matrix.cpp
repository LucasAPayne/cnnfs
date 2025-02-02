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

    switch(device)
    {
        case Device_CPU:
        {
            result.rows = rows;
            result.cols = cols;
            result.data = (T*)calloc(rows*cols, sizeof(T));
        } break;

        case Device_GPU: result = mat_zeros_gpu<T>(rows, cols); break;

        default: log_invalid_device(device); break;
    }

    ASSERT(result.data);
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

        case Device_GPU: result = mat_full_gpu<T>(rows, cols, fill_value);

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

mat<f32> mat_rand_uniform(f32 min, f32 max, size rows, size cols)
{
    // TODO(lucas): Use set row/col range?
    mat<f32> result = mat_zeros<f32>(rows, cols);

    for (size row = 0; row < rows; ++row)
    {
        for (size col = 0; col < cols; ++col)
           result(row, col) = rand_f32_uniform(min, max);
    }
    
    return result;
}

mat<f32> mat_rand_gauss(f32 mean, f32 std_dev, size rows, size cols)
{
    mat<f32> result = mat_zeros<f32>(rows, cols);

    for (size row = 0; row < rows; ++row)
    {
        for (size col = 0; col < cols; ++col)
           result(row, col) = rand_f32_gauss(mean, std_dev);
    }
    
    return result;
}

mat<f32> mat_rand_gauss_standard(size rows, size cols)
{
    mat<f32> result = mat_zeros<f32>(rows, cols);

    for (size row = 0; row < rows; ++row)
    {
        for (size col = 0; col < cols; ++col)
           result(row, col) = rand_f32_gauss_standard();
    }
    
    return result;
}

template <typename T>
internal void mat_set_row(mat<T> m, vec<T> data, size row)
{
    // To set a row, the vector must have the same number of elements as
    // the matrix has columns.
    ASSERT(m.cols == data.elements);

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
    ASSERT(m.rows == data.elements);

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
    ASSERT(row < m.rows);

    vec<T> result = {};
    result.elements = m.rows;
    result.device = m.device;
    result.data = m.data + row*m.cols;

    return result;
}

template <typename T>
vec<T> mat_get_col(mat<T> m, size col)
{
    ASSERT(col < m.cols);

    vec<T> result = {};
    result.elements = m.rows;
    result.device = m.device;
    result.data = m.data + col;

    return result;
}

template <typename T>
internal void mat_set_row_range(mat<T> m, vec<T> data, size row, size row_offset)
{
    // There must be enough columns after the offset to accommodate the vector.
    ASSERT(m.cols >= data.elements + row_offset);

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
    ASSERT(m.rows >= data.elements + col_offset);

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

/* TODO(lucas): Overload this function to take a matrix or vector,
 * and assert that it must be a column vector with the correct number of elements.
 */
// Stretch a column vector to be a matrix with the same shape as target
// by copying the original column a number of times.
template <typename T>
internal mat<T> mat_stretch_cols(mat<T> orig, mat<T> target)
{
    ASSERT(orig.rows == target.rows);

    mat<T> result = {};
    switch (orig.device)
    {
        case Device_CPU:
        {
            result = mat_zeros<T>(target.rows, target.cols);
            for (size i = 0; i < target.cols; ++i)
            {
                for (size j = 0; j < target.cols; ++j)
                    result(i, j) = orig(i, 0);
            }
        } break;

        case Device_GPU: result = mat_stretch_cols_gpu(orig, target); break;

        default: log_invalid_device(orig.device); break;
    }

    return result;
}

/* TODO(lucas): Overload this function to take a matrix or vector,
 * and assert that it must be a row vector with the correct number of elements.
 */
// Stretch a row vector to be a matrix with the same shape as target
// by copying the original row a number of times.
template <typename T>
internal mat<T> mat_stretch_rows(mat<T> orig, mat<T> target)
{
    ASSERT(orig.cols == target.cols);

    mat<T> result = {};
    switch (orig.device)
    {
        case Device_CPU:
        {
            result = mat_zeros<T>(target.rows, target.cols);
            for (size i = 0; i < target.rows; ++i)
            {
                for (size j = 0; j < target.cols; ++j)
                    result(i, j) = orig(0, j);
            }
        } break;

        case Device_GPU: result = mat_stretch_rows_gpu(orig, target); break;

        default: log_invalid_device(orig.device); break;
    }

    return result;
}

template <typename T>
internal mat<T> mat_add(mat<T> a, mat<T> b, b32 in_place)
{
    ASSERT(a.rows == b.rows);
    ASSERT(a.cols == b.cols);
    ASSERT(a.device == b.device);

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
internal mat<T> mat_stretch_add(mat<T> a, mat<T> b)
{
    /* NOTE(lucas): Matrices must be the same size in both dimensions,
     * or must be the same size in one dimension while one matrix is a row/column vector.
     * In the latter case, add the row/column vector across the matrix
     */
	b32 a_col_vec = a.cols == 1;
	b32 a_row_vec = a.rows == 1;
	b32 b_col_vec = b.cols == 1;
	b32 b_row_vec = b.rows == 1;
	b32 valid_sizes = ((a.rows == b.rows) && (a.cols == b.cols))
	|| ((a_col_vec || b_col_vec) && (a.rows == b.rows))
	|| ((a_row_vec || b_row_vec) && (a.cols == b.cols));
	ASSERT(valid_sizes);
	
    // TODO(lucas): Overwriting here causes a memory leak. Use a temp result instead.
    mat<T> result = {};
	if ((a.rows != b.rows) || (a.cols != b.cols))
	{
		if (b_row_vec)
			result = mat_add(mat_stretch_rows(b, a), a);
		else if (b_col_vec)
			result = mat_add(mat_stretch_cols(b, a), a);
		else if (a_row_vec)
			result = mat_add(mat_stretch_cols(a, b), a);
		else if (a_col_vec)
			result = mat_add(mat_stretch_cols(a, b), a);
	}

    return result;
}

template <typename T>
internal mat<T> mat_scale(mat<T> m, T scale, b32 in_place)
{
    mat<T> result = in_place ? m : mat_copy(m);
    switch(result.device)
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
            for (size row = 0; row < m.rows; ++row)
            {
                for (size col = 0; col < m.cols; ++col)
                    m(row, col) *= scale[row];
            }
        } break;

        case Axis_Cols:
        {
            for (size row = 0; row < m.rows; ++row)
            {
                for (size col = 0; col < m.cols; ++col)
                    m(row, col) *= scale[col];
            }
        } break;

        default: break;
    }
}

template <typename T>
internal void mat_scale(mat<T> m, vec<T> scale, Axis axis)
{
    switch (m.device)
    {
        case Device_CPU: mat_scale_cpu(m, scale, axis); break;
        case Device_GPU: mat_scale_gpu(m, scale, axis); break;
        default: log_invalid_device(m.device); break;
    }
}

template <typename T>
internal mat<T> mat_mul(mat<T> a, mat<T> b)
{
    // For multiplication to be valid, the number of columns in A must equal the number of rows in B
    ASSERT(a.cols == b.rows);
    ASSERT(a.device == b.device);

    mat<T> result = {};
    switch(a.device)
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
    ASSERT(a.rows == b.rows);
    ASSERT(a.cols == b.cols);
    ASSERT(a.device == b.device);

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

        default: log_invalid_device(result.device); break;
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
