#pragma once

#include "types.h"

#include <stdio.h>
#include <cuda_runtime.h>

// NOTE(lucas): CUDA helper functions and macros

struct ThreadLayout
{
    dim3 block_dim;
    dim3 grid_dim;
};

inline ThreadLayout calc_thread_dim(usize rows, usize cols, int block_size_x=16, int block_size_y=16)
{
    ThreadLayout result = {};
    result.block_dim = dim3(block_size_x, block_size_y);
    result.grid_dim = dim3(((int)cols + block_size_x - 1) / block_size_x,
                           ((int)rows + block_size_y - 1) / block_size_y);
    return result;
}

inline ThreadLayout calc_thread_dim(usize elements, int block_size=256)
{
    ThreadLayout result = {};
    result.block_dim = dim3(block_size);
    result.grid_dim = dim3(((int)elements + block_size - 1) / block_size);

    return result;
}

// host code for validating last cuda operation (not kernel launch)
#define cuda_call(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        getchar();
        if (abort)
            exit(code);
    }
}

inline void sync_kernel(void)
{
    cuda_call(cudaGetLastError());
    cuda_call(cudaDeviceSynchronize());
}

// Define atomicAdd wrapper that calls the normal function for supported values.
template <typename T>
inline __device__ T atomic_add(T* address, T val)
{
    static_assert(std::is_same<T, f32>::value || std::is_same<T, f64>::value ||
                  std::is_same<T, i32>::value || std::is_same<T, u32>::value,
                  "atomic_add is currently only supported for f32, f64, i32, and u32.");

    T old = *address;
    atomicAdd(address, val);
    return old;
}

/* Template specialization to handle f64 values, which atomicAdd does not support
 * since f64 addition cannot be atomic. The following is adapted from NVIDIA's CUDA programming guide.
 */
#if __CUDA_ARCH__ < 600
template<>
inline __device__ f64 atomic_add<f64>(f64* address, f64 val)
{
    u64* address_as_u64 = (u64*)address;
    u64 old = *address_as_u64;
    u64 assumed;
    do
    {
        assumed = old;
        old = atomicCAS(address_as_u64, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    return __longlong_as_double(old);
}
#endif

/* Template specialization to handle small (<32 bit) values, as atomicAdd does not allow unaligned memory accesses.
 * Because of shared memory concerns, a simple cast is not enough, and the lower bits need to be masked off,
 * and some other processing needs to occur.
 * Clearly, this extra processing leads to slower adds, so prefer 32-bit integers when possible.
 */
template <>
inline __device__ i8 atomic_add<i8>(i8* address, i8 val)
{
    u32* base_address = (u32*)((usize)address & ~2);
    u32 long_val = ((usize)address & 2) ? ((u32)val << 16) : (u32)val;
    u32 long_old = atomicAdd(base_address, long_val);

    if((usize)address & 2)
        long_old >>= 16;
    else
    {
        u32 overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

        if (overflow)
            atomicSub(base_address, overflow);

        long_old &= 0xffff;
    }

    return (i8)long_old;
}

template <>
inline __device__ u8 atomic_add<u8>(u8* address, u8 val)
{
    return atomic_add((i8*)address, (i8)val);
}

// NOTE(lucas): Function templates won't be compiled without specific instantiation for each type.

// Instantiate a single type
#define INST_TEMPLATE(func, ret_type, T, param_list) template ret_type func<T>param_list;

// Instantiate all types
#define INST_ALL_TYPES(func_macro) \
    func_macro(u8)  \
    func_macro(u16) \
    func_macro(u32) \
    func_macro(u64) \
    func_macro(i8)  \
    func_macro(i16) \
    func_macro(i32) \
    func_macro(i64) \
    func_macro(f32) \
    func_macro(f64) 
