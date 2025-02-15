#include "vector.h"
#include "cuda/vector_cuda.h"
#include "rng.cpp"

template <typename T>
internal vec<T> vec_init(size elements, T* data, Device device)
{
    vec<T> result = {};
    result.elements = elements;
    result.data = data;

    // TODO(lucas): Verify that the device and pointer location match?
    result.device = device;

    return result;
}

template <typename T>
vec<T> vec_zeros(size elements, Device device)
{
    vec<T> result = {};
    switch (device)
    {
        case Device_CPU:
        {
            result.elements = elements;
            result.data = (T*)calloc(elements, sizeof(T));
            ASSERT(result.data, "Vector CPU allocation failed.\n");
        } break;

        case Device_GPU: result = vec_zeros_gpu<T>(elements); break;

        default: log_invalid_device(device); break;
    }
    
    return result;
}

template <typename T>
internal vec<T> vec_full(size elements, T fill_value, Device device)
{
    vec<T> result = {};
    
    switch (device)
    {
        case Device_CPU:
        {
            result = vec_zeros<T>(elements);
            for (size i = 0; i < elements; ++i)
                result[i] = fill_value;
        } break;

        case Device_GPU: result = vec_full_gpu(elements, fill_value); break;

        default: log_invalid_device(device); break;
    }
    
    return result;
}

template <typename T>
internal vec<T> vec_copy(vec<T> v)
{
    vec<T> result = {};
    
    switch (v.device)
    {
        case Device_CPU:
        {
            result = vec_zeros<T>(v.elements);
            for (size i = 0; i < v.elements; ++i)
                result[i] = v[i];
        } break;

        case Device_GPU: result = vec_copy_gpu(v); break;

        default: log_invalid_device(v.device); break;
    }
    
    return result;
}

internal vec<f32> vec_rand_uniform(size n, f32 min, f32 max, Device device)
{
    vec<f32> result = {};
    switch (device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f32>(n);
            for (size i = 0; i < n; ++i)
                result[i] = rand_f32_uniform(min, max);
        } break;

        case Device_GPU: result = vec_rand_uniform_gpu(n, min, max); break;

        default: break;
    }
        
    return result;
}

internal vec<f32> vec_rand_gauss(size n, f32 mean, f32 std_dev, Device device)
{
    vec<f32> result = {};
    switch (device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f32>(n);

            for (size i = 0; i < n; ++i)
                result[i] = rand_f32_gauss(mean, std_dev);
        } break;

        case Device_GPU: result = vec_rand_gauss_gpu(n, mean, std_dev); break;

        default: break;
    }

    return result;
}

internal vec<f32> vec_rand_gauss_standard(size n, Device device)
{
    vec<f32> result = {};

    switch (device)
    {
        case Device_CPU:
        {
            result = vec_zeros<f32>(n);
            for (size i = 0; i < n; ++i)
                result[i] = rand_f32_gauss_standard();
        } break;

        case Device_GPU: result = vec_rand_gauss_standard_gpu(n); break;

        default: break;
    }

    return result;
}

template <typename T>
internal void vec_set_range(vec<T> v, vec<T> data, size offset)
{
    ASSERT(v.elements >= data.elements + offset, "Not enough elements in vector after offset to accommodate the new data.\n");
    ASSERT(v.device == data.device, "The vectors must be on the same device.\n");

    switch (v.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < data.elements; ++i)
                v[offset + i] = data[i];
        } break;

        case Device_GPU: vec_set_range_gpu(v, data, offset); break;

        default: log_invalid_device(v.device); break;
    }
}

template <typename T>
internal void vec_print(vec<T> v)
{
    b32 was_on_gpu = v.device == Device_GPU;
    if (was_on_gpu)
        vec_to(&v, Device_CPU);

    printf("[");
    for (size i = 0; i < v.elements; ++i)
    {
        if (i != 0) printf(", ");

        if constexpr (std::is_same_v<T, u8>)
            printf("%hhu", v[i]);
        else if constexpr (std::is_same_v<T, u16>)
            printf("%hu", v[i]);
        else if constexpr (std::is_same_v<T, u32>)
            printf("%u", v[i]);
        else if constexpr (std::is_same_v<T, u64>)
            printf("%llu", v[i]);
        else if constexpr (std::is_same_v<T, i8>)
            printf("%hhd", v[i]);
        else if constexpr (std::is_same_v<T, i16>)
            printf("%hd", v[i]);
        else if constexpr (std::is_same_v<T, i32>)
            printf("%d", v[i]);
        else if constexpr (std::is_same_v<T, i64>)
            printf("%lld", v[i]);
        else if constexpr (std::is_same_v<T, f32>)
            printf("%f", v[i]);
        else if constexpr (std::is_same_v<T, f64>)
            printf("%f", v[i]);
    }
    printf("]\n");

    if (was_on_gpu)
        vec_to(&v, Device_GPU);
}

template <typename T>
internal vec<T> vec_add(vec<T> a, vec<T> b, b32 in_place)
{
    ASSERT(a.elements == b.elements, "Vector addition requires the vectors to be the same size.\n");
    ASSERT(a.device == b.device, "The vectors must be on the same device.\n");

    vec<T> result = in_place ? a : vec_copy(a);

    switch (a.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < a.elements; ++i)
                result[i] += b[i];
        } break;

        case Device_GPU: vec_add_gpu(result, b); break;

        default: log_invalid_device(a.device); break;
    }

    return result;
}

template <typename T>
internal vec<T> vec_scale(vec<T> v, T c, b32 in_place)
{
    vec<T> result = in_place ? v : vec_copy(v);

    switch (v.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
                result[i] *= c;
        } break;

        case Device_GPU: vec_scale_gpu(result, c); break;

        default: log_invalid_device(v.device); break;
    }

    return result;
}

template <typename T>
internal vec<T> vec_scale_inv(vec<T> v, T c, b32 in_place)
{
    vec<T> result = in_place ? v : vec_copy(v);

    switch (v.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
            {
                if (result[i] != (T)0)
                    result[i] = (T)c / result[i];
            }
        } break;

        case Device_GPU: result = vec_scale_inv_gpu(result, c); break;

        default: log_invalid_device(v.device); break;
    }

    return result;
}

template <typename T>
internal vec<T> vec_reciprocal(vec<T> v, b32 in_place)
{
    vec<T> result = vec_scale_inv(v, (T)1, in_place);
    return result;
}

template <typename T>
internal vec<T> vec_had(vec<T> a, vec<T> b, b32 in_place)
{
    ASSERT(a.elements == b.elements, "The Hadamard product requires the vectors to be the same size.\n");
    ASSERT(a.device == b.device, "The vectors must be on the same device.\n");

    vec<T> result = in_place ? a : vec_copy(a);

    switch (a.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < result.elements; ++i)
                result[i] *= b[i];
        } break;

        case Device_GPU: vec_had_gpu(result, b); break;

        default: log_invalid_device(a.device); break;
    }

    return result;
}

template <typename T>
internal T vec_sum(vec<T> v)
{
    T result = 0;
    switch (v.device)
    {
        case Device_CPU:
        {
            for (size i = 0; i < v.elements; ++i)
                result += v[i];
        } break;

        case Device_GPU: vec_sum_gpu(v); break;

        default: log_invalid_device(v.device); break;
    }

    return result;
}
