cmake_minimum_required(VERSION 3.10)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

project(cnnfs)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCE_DIR}/src/main.c
    ${PROJECT_SOURCE_DIR}/src/datasets.c
    ${PROJECT_SOURCE_DIR}/src/dense_layer.c
    ${PROJECT_SOURCE_DIR}/src/math/cnnfs_math.c
    ${PROJECT_SOURCE_DIR}/src/math/matrix/mat_f32.c
    ${PROJECT_SOURCE_DIR}/src/math/matrix/mat_u8.c
    ${PROJECT_SOURCE_DIR}/src/math/vector/vec_f32.c
    ${PROJECT_SOURCE_DIR}/src/math/vector/vec_u8.c
    ${PROJECT_SOURCE_DIR}/src/util/rng.c)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/src/datasets
    ${PROJECT_SOURCE_DIR}/src/math)

set_target_properties(${PROJECT_NAME} PROPERTIES  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
if (MSVC)
    # C4100 and C4189 are about unused parameters/variables and are not particularly useful
    # /Oi: Generate intrinsic functions
    target_compile_options(${PROJECT_NAME} PRIVATE /Oi /WX /W4 /wd4100 /wd4189 /wd4201 /wd4005)
endif()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    # In debug mode, enable complete debug information (/Zi) and disable optimization (/Od)
    target_compile_options(${PROJECT_NAME} PRIVATE /Zi /Od)
    target_compile_definitions(${PROJECT_NAME} PRIVATE CNNFS_DEBUG)
endif()
