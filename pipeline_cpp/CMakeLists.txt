cmake_minimum_required(VERSION 3.18)
project(remaster_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Paths to dependencies (adjust if your layout differs)
# ---------------------------------------------------------------------------

# NVIDIA Video SDK samples (in-repo)
set(VIDEO_SDK_DIR "${CMAKE_SOURCE_DIR}/../reference-code/video-sdk-samples/Samples")
set(VIDEO_SDK_NVCODEC "${VIDEO_SDK_DIR}/NvCodec")
set(VIDEO_SDK_UTILS "${VIDEO_SDK_DIR}/Utils")

# CUDA
find_package(CUDAToolkit REQUIRED)

# TensorRT -- headers from GitHub, DLLs from vs-mlrt bundle
# We generate .lib import libraries from DLLs at configure time.
set(TRT_DLL_DIR "${CMAKE_SOURCE_DIR}/../tools/vs/vs-plugins/vsmlrt-cuda")
set(TRT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../tools/_downloads/TensorRT-10.6.0/include")

if(NOT EXISTS "${TRT_INCLUDE_DIR}/NvInfer.h")
    message(FATAL_ERROR
        "TensorRT headers not found at ${TRT_INCLUDE_DIR}.\n"
        "Download from: https://github.com/NVIDIA/TensorRT/archive/refs/tags/v10.6.0.zip\n"
        "Extract include/ to tools/_downloads/TensorRT-10.6.0/include/")
endif()

# Generate .lib from nvinfer_10.dll using MSVC tools
set(TRT_LIB_DIR "${CMAKE_BINARY_DIR}/trt_libs")
file(MAKE_DIRECTORY "${TRT_LIB_DIR}")

foreach(DLL_NAME nvinfer_10 nvonnxparser_10)
    set(DEF_FILE "${TRT_LIB_DIR}/${DLL_NAME}.def")
    set(LIB_FILE "${TRT_LIB_DIR}/${DLL_NAME}.lib")
    if(NOT EXISTS "${LIB_FILE}")
        message(STATUS "Generating ${DLL_NAME}.lib from ${DLL_NAME}.dll...")
        # Generate .def file from DLL exports
        execute_process(
            COMMAND dumpbin /exports "${TRT_DLL_DIR}/${DLL_NAME}.dll"
            OUTPUT_VARIABLE EXPORTS_OUTPUT
        )
        # Write minimal .def
        file(WRITE "${DEF_FILE}" "LIBRARY ${DLL_NAME}\nEXPORTS\n")
        string(REPLACE "\n" ";" EXPORT_LINES "${EXPORTS_OUTPUT}")
        set(FOUND_EXPORTS FALSE)
        foreach(LINE ${EXPORT_LINES})
            if(LINE MATCHES "^[ ]+[0-9]+[ ]+[0-9A-Fa-f]+[ ]+[0-9A-Fa-f]+[ ]+([^ ]+)")
                file(APPEND "${DEF_FILE}" "  ${CMAKE_MATCH_1}\n")
                set(FOUND_EXPORTS TRUE)
            endif()
        endforeach()
        # Generate .lib
        execute_process(
            COMMAND lib /def:"${DEF_FILE}" /out:"${LIB_FILE}" /machine:x64
            RESULT_VARIABLE LIB_RESULT
        )
        if(NOT LIB_RESULT EQUAL 0)
            message(WARNING "Failed to generate ${LIB_FILE}")
        endif()
    endif()
endforeach()

set(TRT_NVINFER_LIB "${TRT_LIB_DIR}/nvinfer_10.lib")
set(TRT_NVONNXPARSER_LIB "${TRT_LIB_DIR}/nvonnxparser_10.lib")

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib:     ${TRT_NVINFER_LIB}")

# FFmpeg (avcodec, avformat, avutil)
# On Windows, typically from a shared build (e.g., gyan.dev or BtbN builds)
find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h
    PATHS
        "$ENV{FFMPEG_ROOT}"
        "C:/ffmpeg"
        "C:/Program Files/ffmpeg"
    PATH_SUFFIXES include
)

find_library(AVFORMAT_LIB avformat
    PATHS "$ENV{FFMPEG_ROOT}" "C:/ffmpeg" "C:/Program Files/ffmpeg"
    PATH_SUFFIXES lib lib64 lib/x64
)
find_library(AVCODEC_LIB avcodec
    PATHS "$ENV{FFMPEG_ROOT}" "C:/ffmpeg" "C:/Program Files/ffmpeg"
    PATH_SUFFIXES lib lib64 lib/x64
)
find_library(AVUTIL_LIB avutil
    PATHS "$ENV{FFMPEG_ROOT}" "C:/ffmpeg" "C:/Program Files/ffmpeg"
    PATH_SUFFIXES lib lib64 lib/x64
)

if(NOT FFMPEG_INCLUDE_DIR OR NOT AVFORMAT_LIB)
    message(FATAL_ERROR
        "Could not find FFmpeg. Set FFMPEG_ROOT env var.\n"
        "Need: libavformat, libavcodec, libavutil")
endif()

message(STATUS "FFmpeg include: ${FFMPEG_INCLUDE_DIR}")
message(STATUS "FFmpeg libs:    ${AVFORMAT_LIB} ${AVCODEC_LIB} ${AVUTIL_LIB}")

# ---------------------------------------------------------------------------
# Video SDK sources (NvDecoder, NvEncoder are not header-only)
# ---------------------------------------------------------------------------
set(VIDEO_SDK_SOURCES
    ${VIDEO_SDK_NVCODEC}/NvDecoder/NvDecoder.cpp
    ${VIDEO_SDK_NVCODEC}/NvEncoder/NvEncoder.cpp
    ${VIDEO_SDK_NVCODEC}/NvEncoder/NvEncoderCuda.cpp
)

# ColorSpace.cu from Utils provides conversion helpers (we include for completeness
# but primarily use our own color_kernels.cu)

# ---------------------------------------------------------------------------
# Main target
# ---------------------------------------------------------------------------
add_executable(remaster_pipeline
    main.cpp
    trt_inference.cpp
    color_kernels.cu
    ${VIDEO_SDK_SOURCES}
)

target_include_directories(remaster_pipeline PRIVATE
    ${CMAKE_SOURCE_DIR}
    # Video SDK includes -- the SDK sources use includes like:
    #   "NvDecoder/NvDecoder.h"  (from NvCodec/)
    #   "NvEncoder/NvEncoderCuda.h"  (from NvCodec/)
    #   "../Utils/NvCodecUtils.h"  (from NvCodec/NvDecoder/)
    ${VIDEO_SDK_NVCODEC}             # resolves "NvDecoder/NvDecoder.h", "NvEncoder/NvEncoder.h"
    ${VIDEO_SDK_NVCODEC}/NvDecoder   # resolves "nvcuvid.h", "cuviddec.h"
    ${VIDEO_SDK_NVCODEC}/NvEncoder   # resolves "nvEncodeAPI.h"
    ${VIDEO_SDK_UTILS}               # resolves "NvCodecUtils.h", "FFmpegDemuxer.h", "Logger.h"
    # TensorRT
    ${TRT_INCLUDE_DIR}
    # FFmpeg
    ${FFMPEG_INCLUDE_DIR}
    # CUDA
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(remaster_pipeline PRIVATE
    # CUDA
    CUDA::cuda_driver
    CUDA::cudart
    # TensorRT
    ${TRT_NVINFER_LIB}
    # FFmpeg
    ${AVFORMAT_LIB}
    ${AVCODEC_LIB}
    ${AVUTIL_LIB}
    # nvcuvid and nvEncodeAPI are loaded dynamically by the SDK on Windows
)

# On Windows, nvcuvid.lib is a stub for the driver's nvcuvid.dll (NVDEC).
# nvEncodeAPI64.dll is loaded at runtime via LoadLibrary (the SDK handles this).
if(WIN32)
    target_link_libraries(remaster_pipeline PRIVATE
        ${VIDEO_SDK_NVCODEC}/Lib/x64/nvcuvid.lib
    )
endif()

# CUDA architecture -- target sm_86 (RTX 3060) and sm_89 (RTX 40xx) for compat
set_target_properties(remaster_pipeline PROPERTIES
    CUDA_ARCHITECTURES "86;89"
)

# MSVC-specific flags
if(MSVC)
    target_compile_options(remaster_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /MP>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
    )
    # FFmpegDemuxer.h uses deprecated APIs, suppress warnings
    target_compile_definitions(remaster_pipeline PRIVATE
        _CRT_SECURE_NO_WARNINGS
        FF_API_INIT_PACKET  # Suppress av_init_packet deprecation
    )
endif()

# Install
install(TARGETS remaster_pipeline RUNTIME DESTINATION bin)
