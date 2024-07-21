cmake_minimum_required(VERSION 3.10)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

project(meta)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.c
                               ${PROJECT_SOURCE_DIR}/src/color.h
                               ${PROJECT_SOURCE_DIR}/src/log.h
                               ${PROJECT_SOURCE_DIR}/src/log.c
                               ${PROJECT_SOURCE_DIR}/src/parser.h
                               ${PROJECT_SOURCE_DIR}/src/types.h)

target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src)

set_target_properties(${PROJECT_NAME} PROPERTIES  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR})
if (MSVC)
    # C4100 and C4189 are about unused parameters/variables and are not particularly useful
    # /Oi: Generate intrinsic functions
    target_compile_options(${PROJECT_NAME} PRIVATE /Oi /WX /W4 /wd4100 /wd4189 /wd4201 /wd4005 /D_CRT_SECURE_NO_WARNINGS)
endif()

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    # In debug mode, enable complete debug information (/Zi) and disable optimization (/Od)
    target_compile_options(${PROJECT_NAME} PRIVATE /Zi /Od)
    target_compile_definitions(${PROJECT_NAME} PRIVATE META_DEBUG)
endif()
