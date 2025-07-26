@echo off

@rem Default to release build
set is_debug=0

@rem Check first argument
if /i "%1" EQU "debug" (
    set is_debug=1
    shift
)

@REM Check second argument (or first if debug was not provided)
if /i "%1" EQU "profile" (
    SET cmake_flags="-DPROFILER=1"
)

if not defined CUDA_PATH (
    echo CUDA_PATH is not defined!
    exit /b 1
)

@rem /Oi generates intrinsic functions
@rem /FC displays the full path to files in error messages
@rem /wd2505 disables the warning about unreferenced functions with internal linkage (not useful in unity/STU builds)
@rem /we5262: Treat implicit fallthrough as an error
set compiler_common_flags=/nologo /std:c++20 /Oi /FC /WX /W4 /we5262 /wd4505 /D_CRT_SECURE_NO_WARNINGS

@rem /Zi generates a PDB
@rem /Od disables optimization
@rem /MTd uses the debug multithreaded C library
@rem /MD[d] is the multithreaded DLL [debug] version of the CRT
set compiler_debug_flags=/DCNNFS_DEBUG /Zi /Od /MDd /RTC1 /GS /sdl /fsanitize=address
set compiler_release_flags=/O2 /MD

set include_dirs=/I..\src /I..\src\util /I..\src\math
set cuda_include_dirs=-I ../src -I ../src/util -I ../src/math
set output_names=/Focnnfs.obj /Fecnnfs.exe /Fmcnnfs.map

set linker_flags=/link /opt:ref /incremental:no cudart.lib cnnfs_cuda.lib /LIBPATH:"%CUDA_PATH%\lib\x64"

@rem Ignore same warnings as CPu side
@rem -L sets a path for the linker to search for libraries like cudart.lib
set cuda_common_flags=-Xcompiler="/wd4505 /wd4100" -L"%CUDA_PATH%\lib\x64"

@rem -G generates GPU debug info, and -g generates CPU debug info
set cuda_debug_flags=-G -g -Xcompiler=/MDd

@rem -DNDEBUG disables assert statements in C/C++
set cuda_release_flags=-O3 -DNDEBUG -Xcompiler="/O2 /MD"

if "%is_debug%"=="1" (
    set compiler_flags=%compiler_common_flags% %compiler_debug_flags%
    set cuda_compiler_flags=%cuda_common_flags% %cuda_debug_flags%
) else (
    set compiler_flags=%compiler_common_flags% %compiler_release_flags%
    set cuda_compiler_flags=%cuda_common_flags% %cuda_release_flags%
)

if not exist build mkdir build
pushd build
nvcc %cuda_compiler_flags% %cuda_include_dirs% -c ..\src\math\cuda\cnnfs_cuda.cu -o cnnfs_cuda.obj
lib /nologo /OUT:cnnfs_cuda.lib cnnfs_cuda.obj
cl %compiler_flags% %include_dirs% ..\src\main.cpp %output_names% %linker_flags%
popd
