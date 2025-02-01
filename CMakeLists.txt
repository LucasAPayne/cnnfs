cmake_minimum_required(VERSION 3.24)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

if (MSVC)
    # C4100 and C4189 are about unused parameters/variables and are not particularly useful.
    # Compiling with nvcc generates warnings C4100 and C4505 for unused formal parameters
    # and unreferenced local functions, so these are disabled for CUDA.
    # /Oi: Generate intrinsic functions
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Oi /WX /W4 /wd4100 /wd4189 /wd4201 /wd4005 /D_CRT_SECURE_NO_WARNINGS>
        $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler="/wd4505">
    )
    target_link_options(${PROJECT_NAME} PRIVATE $<DEVICE_LINK:-Xcompiler /wd4100>)
endif()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    # TODO(lucas): These compile options are also specific to MSVC
    # In debug mode, enable complete debug information (/Zi) and disable optimization (/Od)
    target_compile_options(${PROJECT_NAME} PRIVATE /Zi /Od /MTd)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CNNFS_DEBUG)
endif()
