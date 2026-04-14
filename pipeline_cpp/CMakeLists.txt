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

# ---------------------------------------------------------------------------
# TensorRT
# ---------------------------------------------------------------------------
if(WIN32)
    # Windows: headers from GitHub, DLLs from vs-mlrt bundle.
    # We generate .lib import libraries from DLLs at configure time.
    set(TRT_DLL_DIR "${CMAKE_SOURCE_DIR}/../tools/vs/vs-plugins/vsmlrt-cuda")
    set(TRT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/../tools/_downloads/TensorRT-10.16.0/include")

    if(NOT EXISTS "${TRT_INCLUDE_DIR}/NvInfer.h")
        message(FATAL_ERROR
            "TensorRT headers not found at ${TRT_INCLUDE_DIR}.\n"
            "Download from: https://github.com/NVIDIA/TensorRT/archive/refs/tags/v10.16.zip\n"
            "Extract include/ to tools/_downloads/TensorRT-10.16.0/include/")
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
            # Generate .lib (no quotes around /def: and /out: args - CMake handles it)
            execute_process(
                COMMAND lib /def:${DEF_FILE} /out:${LIB_FILE} /machine:x64
                RESULT_VARIABLE LIB_RESULT
            )
            if(NOT LIB_RESULT EQUAL 0)
                message(WARNING "Failed to generate ${LIB_FILE}")
            endif()
        endif()
    endforeach()

    set(TRT_NVINFER_LIB "${TRT_LIB_DIR}/nvinfer_10.lib")
    set(TRT_NVONNXPARSER_LIB "${TRT_LIB_DIR}/nvonnxparser_10.lib")
else()
    # Linux: find TensorRT in standard paths (system install or TENSORRT_ROOT)
    find_path(TRT_INCLUDE_DIR NvInfer.h
        PATHS
            $ENV{TENSORRT_ROOT}/include
            /usr/include/x86_64-linux-gnu
            /usr/include
            /usr/local/include
    )
    find_library(TRT_NVINFER_LIB nvinfer
        PATHS
            $ENV{TENSORRT_ROOT}/lib
            /usr/lib/x86_64-linux-gnu
            /usr/lib
            /usr/local/lib
    )
    if(NOT TRT_INCLUDE_DIR OR NOT TRT_NVINFER_LIB)
        message(FATAL_ERROR
            "TensorRT not found. Set TENSORRT_ROOT env var or install TensorRT.\n"
            "Need: NvInfer.h and libnvinfer.so")
    endif()
endif()

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib:     ${TRT_NVINFER_LIB}")

# ---------------------------------------------------------------------------
# FFmpeg (avcodec, avformat, avutil)
# ---------------------------------------------------------------------------
if(WIN32)
    set(FFMPEG_SEARCH_PATHS
        "${CMAKE_SOURCE_DIR}/../tools/_downloads/ffmpeg-n7.1-latest-win64-gpl-shared-7.1"
        "$ENV{FFMPEG_ROOT}"
        "C:/ffmpeg"
        "C:/Program Files/ffmpeg"
    )
else()
    set(FFMPEG_SEARCH_PATHS
        "$ENV{FFMPEG_ROOT}"
        "/usr"
        "/usr/local"
    )
endif()

find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h
    PATHS ${FFMPEG_SEARCH_PATHS}
    PATH_SUFFIXES include include/x86_64-linux-gnu
)

find_library(AVFORMAT_LIB avformat
    PATHS ${FFMPEG_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu
)
find_library(AVCODEC_LIB avcodec
    PATHS ${FFMPEG_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu
)
find_library(AVUTIL_LIB avutil
    PATHS ${FFMPEG_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x64 lib/x86_64-linux-gnu
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
# NvEncoder sources are local copies with updated SDK 13.0 nvEncodeAPI.h
# (the submodule has SDK 8.1 headers which are rejected by modern drivers)
set(VIDEO_SDK_SOURCES
    ${VIDEO_SDK_NVCODEC}/NvDecoder/NvDecoder.cpp
    ${CMAKE_SOURCE_DIR}/nvenc/NvEncoder.cpp
    ${CMAKE_SOURCE_DIR}/nvenc/NvEncoderCuda.cpp
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
    # Local NvEncoder with updated SDK 13.0 headers
    ${CMAKE_SOURCE_DIR}/nvenc                # resolves "NvEncoder.h", "NvEncoderCuda.h"
    ${CMAKE_SOURCE_DIR}/include              # resolves "nvEncodeAPI.h" (SDK 13.0)
    # Video SDK includes -- NvDecoder still uses the submodule
    ${VIDEO_SDK_NVCODEC}             # resolves "NvDecoder/NvDecoder.h"
    ${VIDEO_SDK_NVCODEC}/NvDecoder   # resolves "nvcuvid.h", "cuviddec.h"
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
)

# Platform-specific linking
if(WIN32)
    # nvcuvid.lib is a stub for the driver's nvcuvid.dll (NVDEC).
    # nvEncodeAPI64.dll is loaded at runtime via LoadLibrary (the SDK handles this).
    target_link_libraries(remaster_pipeline PRIVATE
        ${VIDEO_SDK_NVCODEC}/Lib/x64/nvcuvid.lib
    )
else()
    # Linux: link against SDK stub for nvcuvid (real driver lib injected at runtime).
    # NvEncoder uses dlopen("libnvidia-encode.so.1") -- only needs -ldl.
    set(NVCUVID_STUBS_DIR "${VIDEO_SDK_NVCODEC}/Lib/linux/stubs/x86_64")
    find_library(NVCUVID_LIB nvcuvid
        PATHS ${NVCUVID_STUBS_DIR}
        NO_DEFAULT_PATH
    )
    if(NOT NVCUVID_LIB)
        message(FATAL_ERROR "libnvcuvid.so stub not found at ${NVCUVID_STUBS_DIR}")
    endif()
    target_link_libraries(remaster_pipeline PRIVATE
        ${NVCUVID_LIB}
        ${CMAKE_DL_LIBS}  # -ldl for NvEncoder's dlopen
    )
endif()

# CUDA architecture -- target common GPU architectures
if(DEFINED ENV{CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CUDA_ARCHITECTURES})
elseif(WIN32)
    # Local dev: RTX 3060 (sm_86) and RTX 40xx (sm_89)
    set(CMAKE_CUDA_ARCHITECTURES "86;89")
else()
    # Docker/Linux: broad compatibility (Turing through Blackwell)
    # CUDA 13.x dropped compute_60/70 (Pascal/Volta)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;100;120")
endif()
set_target_properties(remaster_pipeline PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

# Compiler flags
if(MSVC)
    target_compile_options(remaster_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3 /MP>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3>
    )
    target_compile_definitions(remaster_pipeline PRIVATE
        _CRT_SECURE_NO_WARNINGS
    )
else()
    target_compile_options(remaster_pipeline PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -O2 -pthread>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-O2>
    )
    target_link_options(remaster_pipeline PRIVATE -pthread)
endif()

# Install
install(TARGETS remaster_pipeline RUNTIME DESTINATION bin)
