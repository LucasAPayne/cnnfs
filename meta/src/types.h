#pragma once

#include <stddef.h>
#include <stdint.h>

// NOTE(lucas): Define additional keywords for different uses of static for additional clarity
// internal: for static functions used only in the translation unit where they are defined
// local_persist: for static variables in functions whose values persist across multiple function calls
// global_variable: for static variables declared at file scope to be used as a global variable
#define internal static
#define persist  static
#define global   static

typedef uint32_t b32;
#define false 0
#define true 1

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef size_t    usize;
typedef ptrdiff_t size;

typedef int8_t  i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef float f32;
typedef double f64;

typedef unsigned char ubyte;

#define countof(a)  (size)(sizeof(a) / sizeof(*(a)))
#define lengthof(a) (countof(a) - 1)

#define ASSERT(expression) if(!(expression)) {*(int *)0 = 0;}
