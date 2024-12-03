#include "vector.h"
#include "cuda/vector_cuda.h"
#include "rng.cpp"

template <typename T>
internal vec<T> vec_init(usize elements, T* data, Device device)
{
    vec<T> result = {};
    result.elements = elements;
    result.data = data;

    // TODO(lucas): Verify that the device and pointer location match?
    result.device = device;

    return result;
}

template <typename T>
vec<T> vec_zeros(usize elements, Device device)
{
    vec<T> result = {};
    switch (device)
    {
        case DEVICE_CPU:
        {
            result.elements = elements;
            result.data = (T*)calloc(elements, sizeof(T));
        } break;

        case DEVICE_GPU: result = vec_zeros_gpu<T>(elements); break;

        default: break;
    }
    
    ASSERT(result.data);
    return result;
}

template <typename T>
internal vec<T> vec_full(usize elements, T fill_value, Device device)
{
    vec<T> result = {};
    
    switch (device)
    {
        case DEVICE_CPU:
        {
            result = vec_zeros<T>(elements);
            for (usize i = 0; i < elements; ++i)
                result.data[i] = fill_value;
        } break;

        case DEVICE_GPU: result = vec_full_gpu(elements, fill_value); break;

        default: break;
    }
    
    return result;
}

vec<f32> vec_rand_uniform(f32 min, f32 max, usize n)
{
    vec<f32> result = vec_zeros<f32>(n);

    for (usize i = 0; i < n; ++i)
        result.data[i] = rand_f32_uniform(min, max);

    return result;
}

vec<f32> vec_rand_gauss(f32 mean, f32 std_dev, usize n)
{
    vec<f32> result = vec_zeros<f32>(n);

    for (usize i = 0; i < n; ++i)
        result.data[i] = rand_f32_gauss(mean, std_dev);

    return result;
}

vec<f32> vec_rand_gauss_standard(usize n)
{
    vec<f32> result = vec_zeros<f32>(n);

    for (usize i = 0; i < n; ++i)
        result.data[i] = rand_f32_gauss_standard();

    return result;
}

template <typename T>
internal void vec_set_range(vec<T> v, vec<T> data, usize offset)
{
    // Ensure there is enough room in the vector
    ASSERT(v.elements >= data.elements + offset);
    ASSERT(v.device == data.device);

    switch(v.device)
    {
        case DEVICE_CPU:
        {
            for (usize i = 0; i < data.elements; ++i)
                v.data[offset + i] = data.data[i];
        } break;

        case DEVICE_GPU: vec_set_range_gpu(v, data, offset); break;

        default: break;
    }
}

template <typename T>
internal void vec_print(vec<T> v)
{
    b32 was_on_gpu = v.device == DEVICE_GPU;
    if (was_on_gpu)
        vec_to(&v, DEVICE_CPU);

    printf("[");
    for (usize i = 0; i < v.elements; ++i)
    {
        if (i != 0) printf(", ");

        if constexpr (std::is_same_v<T, u8>)
            printf("%hhu", v.data[i]);
        else if constexpr (std::is_same_v<T, u16>)
            printf("%hu", v.data[i]);
        else if constexpr (std::is_same_v<T, u32>)
            printf("%u", v.data[i]);
        else if constexpr (std::is_same_v<T, u64>)
            printf("%llu", v.data[i]);
        else if constexpr (std::is_same_v<T, i8>)
            printf("%hhd", v.data[i]);
        else if constexpr (std::is_same_v<T, i16>)
            printf("%hd", v.data[i]);
        else if constexpr (std::is_same_v<T, i32>)
            printf("%d", v.data[i]);
        else if constexpr (std::is_same_v<T, i64>)
            printf("%lld", v.data[i]);
        else if constexpr (std::is_same_v<T, f32>)
            printf("%f", v.data[i]);
        else if constexpr (std::is_same_v<T, f64>)
            printf("%f", v.data[i]);
    }
    printf("]\n");

    if (was_on_gpu)
        vec_to(&v, DEVICE_GPU);
}

template <typename T>
internal void vec_add(vec<T> a, vec<T> b)
{
    ASSERT(a.elements == b.elements);
    ASSERT(a.device == b.device);

    switch (a.device)
    {
        case DEVICE_CPU:
        {
            for (usize i = 0; i < a.elements; ++i)
                a.data[i] += b.data[i];
        } break;

        case DEVICE_GPU: vec_add_gpu(a, b); break;

        default: break;
    }
}

template <typename T>
internal void vec_scale(vec<T> v, T c)
{
    switch (v.device)
    {
        case DEVICE_CPU:
        {
            for (usize i = 0; i < v.elements; ++i)
                v.data[i] *= c;
        } break;

        case DEVICE_GPU: vec_scale_gpu(v, c); break;

        default: break;
    }
}

template <typename T>
internal vec<T> vec_reciprocal(vec<T> v)
{
    switch (v.device)
    {
        case DEVICE_CPU:
        {
            for (usize i = 0; i < v.elements; ++i)
                v.data[i] = (T)1 / v.data[i];
        } break;

        case DEVICE_GPU: vec_reciprocal_gpu(v); break;

        default: break;
    }

    return v;
}

template <typename T>
internal void vec_had(vec<T> a, vec<T> b)
{
    ASSERT(a.elements == b.elements);
    ASSERT(a.device == b.device);

    switch (a.device)
    {
        case DEVICE_CPU:
        {
            for (usize i = 0; i < a.elements; ++i)
                a.data[i] *= b.data[i];
        } break;

        case DEVICE_GPU: vec_had_gpu(a, b); break;

        default: break;
    }
}

template <typename T>
internal T vec_sum(vec<T> v)
{
    T result = 0;
    switch (v.device)
    {
        case DEVICE_CPU:
        {
            for (usize i = 0; i < v.elements; ++i)
                result += v.data[i];
        } break;

        case DEVICE_GPU: vec_sum_gpu(v); break;

        default: break;
    }

    return result;
}
