@echo off
REM Build the C++ TRT video pipeline
REM Run from: pipeline_cpp\build.bat

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=amd64

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
set PATH=%CUDA_PATH%\bin;%PATH%
set TRT_ROOT=C:\Users\sean\src\upscale-experiment\tools\vs\vs-plugins\vsmlrt-cuda

cd /d "%~dp0"
if exist build\CMakeCache.txt del build\CMakeCache.txt
if not exist build mkdir build
cd build

cmake -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe" ^
  -DCMAKE_C_COMPILER=cl ^
  -DCMAKE_CXX_COMPILER=cl ^
  -DTENSORRT_ROOT="%TRT_ROOT%" ^
  ..

if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo Build successful!
echo Binary: %cd%\remaster_trt.exe
pause
