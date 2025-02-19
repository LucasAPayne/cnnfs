@echo off

SET cmake_flags=""
if /i "%1" EQU "profile" set cmake_flags="-DPROFILER=1"

IF NOT EXIST build mkdir build
pushd build
cmake .. -DCMAKE_CXX_FLAGS=%cmake_flags%
cmake --build . --config Debug -j
popd
