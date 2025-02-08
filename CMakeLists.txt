cmake_minimum_required(VERSION 3.24)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_VERBOSE_MAKEFILE ON)

project(cnnfs LANGUAGES CXX CUDA)

set(C_SRC ${PROJECT_SOURCE_DIR}/src/main.cpp)
set(CUDA_SRC
    ${PROJECT_SOURCE_DIR}/src/math/cuda/activation.cu
    ${PROJECT_SOURCE_DIR}/src/math/cuda/cnnfs_math.cu
    ${PROJECT_SOURCE_DIR}/src/math/cuda/matrix.cu
    ${PROJECT_SOURCE_DIR}/src/math/cuda/vector.cu
)

add_executable(${PROJECT_NAME} ${C_SRC} ${CUDA_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/util
    ${PROJECT_SOURCE_DIR}/src/math
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

if(CMAKE_CONFIGURATION_TYPES)
    set(BUILD_TYPE "$<CONFIG>")
else()
    set(BUILD_TYPE "${CMAKE_BUILD_TYPE}")
endif()

# Disable RTC flag for the whole project.
# It needs to be disabled for release builds, and it is enabled explicitly for debug builds.
foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_RELEASE)
    STRING(REGEX REPLACE "[/|-]RTC[^ ]*" "" ${flag_var} "${${flag_var}}")
endforeach(flag_var)

if(MSVC)
    # C4100 and C4189 are about unused parameters/variables and are not particularly useful.
    # /we5262: Treat implicit fallthrough as an error.
    # /Oi: Generate intrinsic functions.
    set(COMPILER_FLAGS_COMMON /Oi /WX /W4 /wd4100 /wd4189 /wd4201 /wd4005 /we5262 /D_CRT_SECURE_NO_WARNINGS)

    # In debug mode, enable complete debug information (/Zi) and disable optimization (/Od)
    # Specify the DLL multithreaded debug version of CRT (MDd) and perform all available run-time error checks (RTC1)
    set(COMPILER_FLAGS_DEBUG /Zi /Od /RTC1)

    # In release mode, enable full optimization (/O2)
    set(COMPILER_FLAGS_RELEASE /O2)

    # Ignore all CRT libraries except MDd (DLL multithreaded debug) in debug mode
    # and MD (DLL multithreaded) in release mode
    set(LINKER_FLAGS_DEBUG /NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrt.lib)
    set(LINKER_FLAGS_RELEASE /NODEFAULTLIB:libcmt.lib /NODEFAULTLIB:libcmtd.lib /NODEFAULTLIB:msvcrtd.lib)

    # Compiling with nvcc generates warnings C4100 and C4505 for unused formal parameters
    # and unreferenced local functions, so these are disabled for CUDA.
    set(CUDA_FLAGS_COMMON -Xcompiler=/wd4505)
    set(CUDA_FLAGS_DEBUG -G -g)
    set(CUDA_FLAGS_RELEASE -O3 -DNDEBUG -Xcompiler=/O2)

    set(CUDA_LINKER_FLAGS_COMMON -Xcompiler=/wd4100)

    target_compile_definitions(${PROJECT_NAME} PRIVATE $<$<CONFIG:Debug>:CNNFS_DEBUG>)

    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${COMPILER_FLAGS_COMMON}>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:${COMPILER_FLAGS_DEBUG}>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:${COMPILER_FLAGS_RELEASE}>>
    )

    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS_COMMON}>
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:${CUDA_FLAGS_DEBUG}>>
        $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Release>:${CUDA_FLAGS_RELEASE}>>
    )

    target_link_options(${PROJECT_NAME} PRIVATE $<DEVICE_LINK:${CUDA_LINKER_FLAGS_COMMON}>)
    
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:${LINKER_FLAGS_DEBUG}>>
        $<$<COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Release>:${LINKER_FLAGS_RELEASE}>>
    )
endif()
