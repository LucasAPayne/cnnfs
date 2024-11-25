cmake_minimum_required(VERSION 3.24)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

project(cnnfs LANGUAGES C CXX CUDA)

set(C_SRC ${PROJECT_SOURCE_DIR}/src/main.c
    ${PROJECT_SOURCE_DIR}/src/datasets.c
    ${PROJECT_SOURCE_DIR}/src/dense_layer.c
    ${PROJECT_SOURCE_DIR}/src/math/cnnfs_math.c
    ${PROJECT_SOURCE_DIR}/src/math/vector.c
    ${PROJECT_SOURCE_DIR}/src/math/matrix.c
    ${PROJECT_SOURCE_DIR}/src/util/rng.c
)

set(CUDA_SRC ${PROJECT_SOURCE_DIR}/src/math/matrix.cu)

add_executable(${PROJECT_NAME} ${C_SRC} ${CUDA_SRC})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/datasets
    ${PROJECT_SOURCE_DIR}/src/math
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

if (MSVC)
    # C4100 and C4189 are about unused parameters/variables and are not particularly useful
    # /Oi: Generate intrinsic functions
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/Oi /WX /W4 /wd4100 /wd4189 /wd4201 /wd4005 /D_CRT_SECURE_NO_WARNINGS>
        $<$<COMPILE_LANGUAGE:CUDA>:>
    )
endif()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    # TODO(lucas): These compile options are also specific to MSVC
    # In debug mode, enable complete debug information (/Zi) and disable optimization (/Od)
    target_compile_options(${PROJECT_NAME} PRIVATE /Zi /Od /MTd)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CNNFS_DEBUG)
endif()
