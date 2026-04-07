// color_kernels.h -- GPU color space conversion between NV12/P010 and planar RGB FP16
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// NV12 (8-bit YUV 4:2:0) -> Planar RGB FP16 [0,1], BT.709 limited range
// Writes (dstWidth * dstHeight * 3) half-precision floats in R,G,B plane order.
// Pixels outside [srcWidth, srcHeight] are zero-padded.
void launchNv12ToRgbFp16(
    const uint8_t* nv12, __half* rgb,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight,
    cudaStream_t stream);

// P010 (10-bit YUV 4:2:0) -> Planar RGB FP16 [0,1], BT.709 limited range
void launchP010ToRgbFp16(
    const uint8_t* p010, __half* rgb,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight,
    cudaStream_t stream);

// Planar RGB FP16 [0,1] -> NV12, BT.709 limited range
// Reads from padded (rgbWidth * rgbHeight) planes, writes (srcWidth * srcHeight) NV12.
void launchRgbFp16ToNv12(
    const __half* rgb, uint8_t* nv12,
    int srcWidth, int srcHeight,
    int rgbWidth, int rgbHeight,
    int dstPitch,
    cudaStream_t stream);

// Planar RGB FP16 [0,1] -> P010, BT.709 limited range
void launchRgbFp16ToP010(
    const __half* rgb, uint8_t* p010,
    int srcWidth, int srcHeight,
    int rgbWidth, int rgbHeight,
    int dstPitch,
    cudaStream_t stream);

// Round up to the nearest multiple of 8 (TRT model requirement)
inline int alignTo8(int val) {
    return (val + 7) & ~7;
}
