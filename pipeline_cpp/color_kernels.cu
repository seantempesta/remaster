// color_kernels.cu -- NV12/P010 <-> planar RGB FP16 conversion kernels (BT.709)
//
// These kernels convert between NVDEC's native NV12/P010 surface format and
// the planar RGB FP16 [0,1] layout expected by our TensorRT inference engine.
// All math uses BT.709 limited-range coefficients.

#include "color_kernels.h"
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------
// BT.709 limited-range constants
//
//   Y  = [16..235]  (8-bit) or [64..940]  (10-bit, stored in upper 10 of 16)
//   UV = [16..240]  (8-bit) or [64..960]  (10-bit)
//
// Forward (YUV -> RGB):
//   R = 1.164*(Y-16) + 1.793*(V-128)
//   G = 1.164*(Y-16) - 0.213*(U-128) - 0.533*(V-128)
//   B = 1.164*(Y-16) + 2.112*(U-128)
//
// Inverse (RGB -> YUV):  (RGB in [0,255])
//   Y  =  0.2126*R + 0.7152*G + 0.0722*B  (full-range)
//   ... we use the standard limited-range matrix below.
// ---------------------------------------------------------------------------

// BT.709 YUV->RGB matrix for limited range
// These incorporate the (Y-16)*255/219 and (UV-128)*255/224 scaling.
__device__ __constant__ float kYuvToRgb[9] = {
    1.164384f,  0.000000f,  1.792741f,   // R
    1.164384f, -0.213249f, -0.532909f,   // G
    1.164384f,  2.112402f,  0.000000f    // B
};

// BT.709 RGB->YUV matrix for limited range
// Maps [0,1] float RGB to [16,235] Y and [16,240] UV (as fractions of 255)
__device__ __constant__ float kRgbToYuv[9] = {
     0.182586f,  0.614231f,  0.062007f,  // Y  (scaled to give 16..235 when input 0..255)
    -0.100644f, -0.338572f,  0.439216f,  // U
     0.439216f, -0.398942f, -0.040274f   // V
};

// ---------------------------------------------------------------------------
// NV12 -> Planar RGB FP16 [0,1]
// ---------------------------------------------------------------------------
// NV12 layout: W*H bytes of Y, then (W*H/2) bytes of interleaved UV
// The decoder may use a pitch (stride) larger than width.
//
// Output: 3 contiguous planes of __half, each (padH * padW), in R,G,B order.
// We write zeros in padding regions.

__global__ void kernel_nv12_to_rgb_fp16(
    const uint8_t* __restrict__ nv12,   // NVDEC frame (device ptr, pitched)
    __half* __restrict__ rgb,            // output: 3 planar FP16 [R,G,B]
    int srcWidth, int srcHeight,
    int srcPitch,                        // NVDEC pitch (bytes per row of Y)
    int dstWidth, int dstHeight          // padded dims (divisible by 8)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;

    int planeSize = dstWidth * dstHeight;
    int dstIdx = y * dstWidth + x;

    if (x >= srcWidth || y >= srcHeight) {
        // Padding region: write zero
        rgb[dstIdx]                = __float2half(0.0f);
        rgb[dstIdx + planeSize]    = __float2half(0.0f);
        rgb[dstIdx + planeSize*2]  = __float2half(0.0f);
        return;
    }

    // Read Y
    float yVal = (float)nv12[y * srcPitch + x];

    // Read UV (subsampled 2x2)
    int uvRow = srcHeight + (y >> 1);
    int uvCol = (x & ~1); // align to even
    float uVal = (float)nv12[uvRow * srcPitch + uvCol];
    float vVal = (float)nv12[uvRow * srcPitch + uvCol + 1];

    // BT.709 limited range -> [0,1]
    float y0 = yVal - 16.0f;
    float u0 = uVal - 128.0f;
    float v0 = vVal - 128.0f;

    float r = kYuvToRgb[0]*y0 + kYuvToRgb[1]*u0 + kYuvToRgb[2]*v0;
    float g = kYuvToRgb[3]*y0 + kYuvToRgb[4]*u0 + kYuvToRgb[5]*v0;
    float b = kYuvToRgb[6]*y0 + kYuvToRgb[7]*u0 + kYuvToRgb[8]*v0;

    // Normalize to [0,1] and clamp
    r = fminf(fmaxf(r / 255.0f, 0.0f), 1.0f);
    g = fminf(fmaxf(g / 255.0f, 0.0f), 1.0f);
    b = fminf(fmaxf(b / 255.0f, 0.0f), 1.0f);

    rgb[dstIdx]                = __float2half(r);
    rgb[dstIdx + planeSize]    = __float2half(g);
    rgb[dstIdx + planeSize*2]  = __float2half(b);
}

// ---------------------------------------------------------------------------
// P010 -> Planar RGB FP16 [0,1]
// ---------------------------------------------------------------------------
// P010 layout: W*H uint16 of Y (10-bit in upper bits), then W*H/2 uint16 UV
// srcPitch is in bytes.

__global__ void kernel_p010_to_rgb_fp16(
    const uint8_t* __restrict__ p010,
    __half* __restrict__ rgb,
    int srcWidth, int srcHeight,
    int srcPitch,
    int dstWidth, int dstHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) return;

    int planeSize = dstWidth * dstHeight;
    int dstIdx = y * dstWidth + x;

    if (x >= srcWidth || y >= srcHeight) {
        rgb[dstIdx]                = __float2half(0.0f);
        rgb[dstIdx + planeSize]    = __float2half(0.0f);
        rgb[dstIdx + planeSize*2]  = __float2half(0.0f);
        return;
    }

    // P010: each sample is uint16, value in upper 10 bits. Divide by 65535 later.
    const uint16_t* yPlane = (const uint16_t*)(p010 + y * srcPitch);
    float yVal = (float)(yPlane[x] >> 6); // shift to 10-bit range [0,1023]

    int uvRow = srcHeight + (y >> 1);
    const uint16_t* uvPlane = (const uint16_t*)(p010 + uvRow * srcPitch);
    int uvCol = (x & ~1);
    float uVal = (float)(uvPlane[uvCol] >> 6);
    float vVal = (float)(uvPlane[uvCol + 1] >> 6);

    // BT.709 10-bit limited range: Y [64,940], UV [64,960]
    float y0 = yVal - 64.0f;
    float u0 = uVal - 512.0f;
    float v0 = vVal - 512.0f;

    // Scale factors for 10-bit: Y range is 876 (940-64), full range 1023
    // Reuse the 8-bit matrix but scale: multiply by (255/219)*(1023/255) = 1023/219
    // Actually simpler: apply the matrix with 10-bit aware scaling
    float scale_y = 1.0f / 876.0f;  // (940-64)
    float scale_uv = 1.0f / 896.0f; // (960-64)

    float r = y0 * scale_y + 1.792741f * v0 * scale_uv;
    float g = y0 * scale_y - 0.213249f * u0 * scale_uv - 0.532909f * v0 * scale_uv;
    float b = y0 * scale_y + 2.112402f * u0 * scale_uv;

    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    rgb[dstIdx]                = __float2half(r);
    rgb[dstIdx + planeSize]    = __float2half(g);
    rgb[dstIdx + planeSize*2]  = __float2half(b);
}

// ---------------------------------------------------------------------------
// Planar RGB FP16 [0,1] -> NV12
// ---------------------------------------------------------------------------
__global__ void kernel_rgb_fp16_to_nv12_luma(
    const __half* __restrict__ rgb,
    uint8_t* __restrict__ nv12,
    int srcWidth, int srcHeight,   // original (unpadded) dims
    int rgbWidth, int rgbHeight,   // padded dims
    int dstPitch                   // NVENC pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= srcWidth || y >= srcHeight) return;

    int planeSize = rgbWidth * rgbHeight;
    int srcIdx = y * rgbWidth + x;

    float r = __half2float(rgb[srcIdx]);
    float g = __half2float(rgb[srcIdx + planeSize]);
    float b = __half2float(rgb[srcIdx + planeSize*2]);

    // Clamp to [0,1]
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    // Scale to [0,255]
    r *= 255.0f; g *= 255.0f; b *= 255.0f;

    // BT.709 limited range
    float yVal = kRgbToYuv[0]*r + kRgbToYuv[1]*g + kRgbToYuv[2]*b + 16.0f;
    nv12[y * dstPitch + x] = (uint8_t)fminf(fmaxf(yVal + 0.5f, 0.0f), 255.0f);
}

__global__ void kernel_rgb_fp16_to_nv12_chroma(
    const __half* __restrict__ rgb,
    uint8_t* __restrict__ nv12,
    int srcWidth, int srcHeight,
    int rgbWidth, int rgbHeight,
    int dstPitch
) {
    // Each thread handles one 2x2 block -> one UV pair
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    int chromaW = srcWidth / 2;
    int chromaH = srcHeight / 2;
    if (cx >= chromaW || cy >= chromaH) return;

    int planeSize = rgbWidth * rgbHeight;
    int px = cx * 2;
    int py = cy * 2;

    // Average 2x2 block
    float rSum = 0, gSum = 0, bSum = 0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int idx = (py+dy) * rgbWidth + (px+dx);
            float rv = fminf(fmaxf(__half2float(rgb[idx]), 0.0f), 1.0f) * 255.0f;
            float gv = fminf(fmaxf(__half2float(rgb[idx + planeSize]), 0.0f), 1.0f) * 255.0f;
            float bv = fminf(fmaxf(__half2float(rgb[idx + planeSize*2]), 0.0f), 1.0f) * 255.0f;
            rSum += rv; gSum += gv; bSum += bv;
        }
    }
    rSum *= 0.25f; gSum *= 0.25f; bSum *= 0.25f;

    float uVal = kRgbToYuv[3]*rSum + kRgbToYuv[4]*gSum + kRgbToYuv[5]*bSum + 128.0f;
    float vVal = kRgbToYuv[6]*rSum + kRgbToYuv[7]*gSum + kRgbToYuv[8]*bSum + 128.0f;

    int uvOffset = srcHeight * dstPitch; // chroma plane starts after luma
    int uvIdx = uvOffset + cy * dstPitch + cx * 2;
    nv12[uvIdx]     = (uint8_t)fminf(fmaxf(uVal + 0.5f, 0.0f), 255.0f);
    nv12[uvIdx + 1] = (uint8_t)fminf(fmaxf(vVal + 0.5f, 0.0f), 255.0f);
}

// ---------------------------------------------------------------------------
// Planar RGB FP16 [0,1] -> P010
// ---------------------------------------------------------------------------
__global__ void kernel_rgb_fp16_to_p010_luma(
    const __half* __restrict__ rgb,
    uint8_t* __restrict__ p010,
    int srcWidth, int srcHeight,
    int rgbWidth, int rgbHeight,
    int dstPitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= srcWidth || y >= srcHeight) return;

    int planeSize = rgbWidth * rgbHeight;
    int srcIdx = y * rgbWidth + x;

    float r = fminf(fmaxf(__half2float(rgb[srcIdx]), 0.0f), 1.0f);
    float g = fminf(fmaxf(__half2float(rgb[srcIdx + planeSize]), 0.0f), 1.0f);
    float b = fminf(fmaxf(__half2float(rgb[srcIdx + planeSize*2]), 0.0f), 1.0f);

    // BT.709 limited range, 10-bit: Y in [64, 940]
    float yVal = (0.2126f*r + 0.7152f*g + 0.0722f*b) * 876.0f + 64.0f;
    uint16_t y10 = (uint16_t)fminf(fmaxf(yVal + 0.5f, 0.0f), 1023.0f);

    // P010: store in upper 10 bits of uint16
    uint16_t* dst = (uint16_t*)(p010 + y * dstPitch);
    dst[x] = y10 << 6;
}

__global__ void kernel_rgb_fp16_to_p010_chroma(
    const __half* __restrict__ rgb,
    uint8_t* __restrict__ p010,
    int srcWidth, int srcHeight,
    int rgbWidth, int rgbHeight,
    int dstPitch
) {
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    int chromaW = srcWidth / 2;
    int chromaH = srcHeight / 2;
    if (cx >= chromaW || cy >= chromaH) return;

    int planeSize = rgbWidth * rgbHeight;
    int px = cx * 2;
    int py = cy * 2;

    float rSum = 0, gSum = 0, bSum = 0;
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int idx = (py+dy) * rgbWidth + (px+dx);
            rSum += fminf(fmaxf(__half2float(rgb[idx]), 0.0f), 1.0f);
            gSum += fminf(fmaxf(__half2float(rgb[idx + planeSize]), 0.0f), 1.0f);
            bSum += fminf(fmaxf(__half2float(rgb[idx + planeSize*2]), 0.0f), 1.0f);
        }
    }
    rSum *= 0.25f; gSum *= 0.25f; bSum *= 0.25f;

    // BT.709 10-bit: U,V in [64, 960]
    float wg = 1.0f - 0.2126f - 0.0722f; // 0.7152
    float uVal = (-0.2126f/1.8556f*rSum - wg/1.8556f*gSum + 0.5f*bSum) * 896.0f + 512.0f;
    float vVal = (0.5f*rSum - wg/1.5748f*gSum - 0.0722f/1.5748f*bSum) * 896.0f + 512.0f;

    uint16_t u10 = (uint16_t)fminf(fmaxf(uVal + 0.5f, 0.0f), 1023.0f);
    uint16_t v10 = (uint16_t)fminf(fmaxf(vVal + 0.5f, 0.0f), 1023.0f);

    int uvOffset = srcHeight * dstPitch;
    uint16_t* dst = (uint16_t*)(p010 + uvOffset + cy * dstPitch);
    dst[cx * 2]     = u10 << 6;
    dst[cx * 2 + 1] = v10 << 6;
}

// ---------------------------------------------------------------------------
// Host-callable wrappers
// ---------------------------------------------------------------------------

void launchNv12ToRgbFp16(
    const uint8_t* nv12, __half* rgb,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((dstWidth + block.x - 1) / block.x,
              (dstHeight + block.y - 1) / block.y);
    kernel_nv12_to_rgb_fp16<<<grid, block, 0, stream>>>(
        nv12, rgb, srcWidth, srcHeight, srcPitch, dstWidth, dstHeight);
}

void launchP010ToRgbFp16(
    const uint8_t* p010, __half* rgb,
    int srcWidth, int srcHeight, int srcPitch,
    int dstWidth, int dstHeight,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((dstWidth + block.x - 1) / block.x,
              (dstHeight + block.y - 1) / block.y);
    kernel_p010_to_rgb_fp16<<<grid, block, 0, stream>>>(
        p010, rgb, srcWidth, srcHeight, srcPitch, dstWidth, dstHeight);
}

void launchRgbFp16ToNv12(
    const __half* rgb, uint8_t* nv12,
    int srcWidth, int srcHeight,
    int rgbWidth, int rgbHeight,
    int dstPitch,
    cudaStream_t stream
) {
    // Luma
    {
        dim3 block(32, 8);
        dim3 grid((srcWidth + block.x - 1) / block.x,
                  (srcHeight + block.y - 1) / block.y);
        kernel_rgb_fp16_to_nv12_luma<<<grid, block, 0, stream>>>(
            rgb, nv12, srcWidth, srcHeight, rgbWidth, rgbHeight, dstPitch);
    }
    // Chroma
    {
        dim3 block(32, 8);
        dim3 grid(((srcWidth/2) + block.x - 1) / block.x,
                  ((srcHeight/2) + block.y - 1) / block.y);
        kernel_rgb_fp16_to_nv12_chroma<<<grid, block, 0, stream>>>(
            rgb, nv12, srcWidth, srcHeight, rgbWidth, rgbHeight, dstPitch);
    }
}

void launchRgbFp16ToP010(
    const __half* rgb, uint8_t* p010,
    int srcWidth, int srcHeight,
    int rgbWidth, int rgbHeight,
    int dstPitch,
    cudaStream_t stream
) {
    // Luma
    {
        dim3 block(32, 8);
        dim3 grid((srcWidth + block.x - 1) / block.x,
                  (srcHeight + block.y - 1) / block.y);
        kernel_rgb_fp16_to_p010_luma<<<grid, block, 0, stream>>>(
            rgb, p010, srcWidth, srcHeight, rgbWidth, rgbHeight, dstPitch);
    }
    // Chroma
    {
        dim3 block(32, 8);
        dim3 grid(((srcWidth/2) + block.x - 1) / block.x,
                  ((srcHeight/2) + block.y - 1) / block.y);
        kernel_rgb_fp16_to_p010_chroma<<<grid, block, 0, stream>>>(
            rgb, p010, srcWidth, srcHeight, rgbWidth, rgbHeight, dstPitch);
    }
}
