@echo off

@REM pushd meta
@REM call build
@REM popd
@REM "meta/meta.exe" src/math/vector.meta src/math/matrix.meta

IF NOT EXIST build mkdir build
pushd build
cmake ..
cmake --build . -j
popd
