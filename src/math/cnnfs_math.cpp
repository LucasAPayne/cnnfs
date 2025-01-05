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

        default: break;
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

        default: break;
    }

    return result;
}

vec<f32> sin_vec(vec<f32> v)
{
    vec<f32> result = {};
    switch (v.device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f32>(v.elements);
            for (size i = 0; i < v.elements; ++i)
                result[i] = sin_f32(v[i]);
        } break;

        case Device_GPU: result = sin_vec_gpu(v); break;

        default: break;
    }

    return result;
}

vec<f64> sin_vec(vec<f64> v)
{
    vec<f64> result = {};
    switch (v.device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f64>(v.elements);
            for (size i = 0; i < v.elements; ++i)
                result[i] = sin_f64(v[i]);
        } break;

        case Device_GPU: result = sin_vec_gpu(v); break;

        default: break;
    }

    return result;
}

vec<f32> cos_vec(vec<f32> v)
{
    vec<f32> result = {};
    switch (v.device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f32>(v.elements);
            for (size i = 0; i < v.elements; ++i)
                result[i] = cos_f32(v[i]);
        } break;

        case Device_GPU: result = cos_vec_gpu(v); break;

        default: break;
    }
    
    return result;
}

vec<f64> cos_vec(vec<f64> v)
{
    vec<f64> result = {};
    switch (v.device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f64>(v.elements);
            for (size i = 0; i < v.elements; ++i)
                v[i] = cos_f64(v[i]);
        } break;

        case Device_GPU: result = cos_vec_gpu(v); break;

        default: break;
    }

    return result;
}

void exp_vec(vec<f32> v)
{
    switch (v.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < v.elements; ++i)
                v[i] = expf(v[i]);
        } break;

        case Device_GPU: exp_vec_gpu(v); break;

        default: break;
    }
}

void exp_vec(vec<f64> v)
{
    switch (v.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < v.elements; ++i)
                v[i] = exp(v[i]);
        } break;

        case Device_GPU: exp_vec_gpu(v); break;

        default: break;
    }
}

void exp_mat(mat<f32> m)
{
    switch (m.device)
    {
        case Device_CPU:
        {
            for (size col = 0; col < m.cols; ++col)
            {
                for (size row = 0; row < m.rows; ++row)
                {
                    f32 val = expf(mat_at(m, row, col));
                    mat_set_val(m, row, col, val);
                }
            }
        } break;

        case Device_GPU: exp_mat_gpu(m); break;

        default: break;
    }
}

void exp_mat(mat<f64> m)
{
    switch (m.device)
    {
        case Device_CPU:
        {
            for (size col = 0; col < m.cols; ++col)
            {
                for (size row = 0; row < m.rows; ++row)
                {
                    f64 val = exp(mat_at(m, row, col));
                    mat_set_val(m, row, col, val);
                }
            }
        } break;

        case Device_GPU: exp_mat_gpu(m); break;

        default: break;
    }
}
