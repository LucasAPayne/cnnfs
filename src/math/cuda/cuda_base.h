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
