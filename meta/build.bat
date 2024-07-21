@echo off

IF NOT EXIST build mkdir build
pushd build
cmake ..
cmake --build . -j
popd
copy /b "build\meta.exe" "meta.exe"
