#include "cnnfs_math.h"
#include "cuda/cnnfs_math_cuda.h"

vec<f32> linspace(f32 x1, f32 x2, size n, Device device)
{
    vec<f32> result = {};
    switch (device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f32>(n, device);
            f32 dx = (x2 - x1) / (n - 1.0f);

            for (size i = 0; i < n; ++i)
                result[i] = x1 + ((f32)i * dx);
        } break;

        case Device_GPU: result = linspace_gpu(x1, x2, n); break;

        default: log_invalid_device(device); break;
    }

    return result;
}

vec<f64> linspace(f64 x1, f64 x2, size n, Device device)
{
    vec<f64> result = {};
    switch (device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f64>(n);
            f64 dx = (x2 - x1) / (n - 1.0f);

            for (size i = 0; i < n; ++i)
                result[i] = x1 + ((f64)i * dx);

            return result;
        } break;

        case Device_GPU: result = linspace_gpu(x1, x2, n); break;

        default: log_invalid_device(device); break;
    }

    return result;
}

vec<f32> sin_vec(vec<f32> v, b32 in_place)
{
    vec<f32> result = in_place ? v : vec_copy(v);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
                result[i] = sin_f32(result[i]);
        } break;

        case Device_GPU: sin_vec_gpu(result); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

vec<f64> sin_vec(vec<f64> v, b32 in_place)
{
    vec<f64> result = in_place ? v : vec_copy(v);
    switch (v.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
                result[i] = sin_f64(result[i]);
        } break;

        case Device_GPU: sin_vec_gpu(result); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

vec<f32> cos_vec(vec<f32> v, b32 in_place)
{
    vec<f32> result = in_place ? v : vec_copy(v);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
                result[i] = cos_f32(result[i]);
        } break;

        case Device_GPU: cos_vec_gpu(result); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

vec<f64> cos_vec(vec<f64> v, b32 in_place)
{
    vec<f64> result = in_place ? v : vec_copy(v);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
                result[i] = cos_f64(result[i]);
        } break;

        case Device_GPU: cos_vec_gpu(result); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

vec<f32> exp_vec(vec<f32> v, b32 in_place)
{
    vec<f32> result = in_place ? v : vec_copy(v);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
                result[i] = expf(result[i]);
        } break;

        case Device_GPU: exp_vec_gpu(result); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

vec<f64> exp_vec(vec<f64> v, b32 in_place)
{
    vec<f64> result = in_place ? v : vec_copy(v);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
                result[i] = exp(result[i]);
        } break;

        case Device_GPU: exp_vec_gpu(result); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

mat<f32> exp_mat(mat<f32> m, b32 in_place)
{
    mat<f32> result = in_place ? m : mat_copy(m);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size col = 0; col < result.cols; ++col)
            {
                for (size row = 0; row < result.rows; ++row)
                {
                    f32 val = expf(result(row, col));
                    result(row, col) = val;
                }
            }
        } break;

        case Device_GPU: exp_mat_gpu(result); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

mat<f64> exp_mat(mat<f64> m, b32 in_place)
{
    mat<f64> result = in_place ? m : mat_copy(m);
    switch (result.device)
    {
        case Device_CPU:
        {
            for (size col = 0; col < result.cols; ++col)
            {
                for (size row = 0; row < result.rows; ++row)
                {
                    f64 val = exp(result(row, col));
                    result(row, col) = val;
                }
            }
        } break;

        case Device_GPU: exp_mat_gpu(result); break;

        default: log_invalid_device(result.device); break;
    }

    return result;
}

// template <typename T>
u32 argmax(vec<f32> v)
{
    u32 max_idx = 0;
    switch (v.device)
    {
        case Device_CPU:
        {
            for (u32 i = 1; i < v.elements; ++i)
            {
                if (v[i] > v[max_idx])
                max_idx = i;
            }
        } break;

        case Device_GPU: max_idx = argmax_gpu(v); break;

        default: log_invalid_device(v.device); break;
    }
    return max_idx;
}

internal vec<u32> argmax_cpu(mat<f32> m, Axis axis)
{
    vec<u32> result;
    switch (axis)
    {
        case Axis_Rows:
        {
            result = vec_zeros<u32>(m.rows, Device_CPU);
            for (size row = 0; row < m.rows; ++row)
            {
                vec<f32> vals = mat_get_row(m, row);
                result[row] = argmax(vals);
            }
        } break;

        case Axis_Cols:
        {
            result = vec_zeros<u32>(m.cols, Device_CPU);
            for (size col = 0; col < m.cols; ++col)
            {
                vec<f32> vals = mat_get_col(m, col);
                result[col] = argmax(vals);
            }
        } break;

        default: log_invalid_axis(axis); break;
    }

    return result;
}

vec<u32> argmax(mat<f32> m, Axis axis)
{
    vec<u32> result;
    switch (m.device)
    {
        case Device_CPU: result = argmax_cpu(m, axis); break;
        case Device_GPU: result = argmax_gpu(m, axis); break;
        default: log_invalid_device(m.device); break;
    }

    return result;
}
