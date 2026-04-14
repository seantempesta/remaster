// color_transfer.cu -- Per-frame color/brightness transfer kernels
//
// Two operations:
// 1. computeChannelStats: parallel reduction to get per-channel mean + std
// 2. applyColorTransfer: element-wise transform to match input color stats
//
// All math in FP32 for reduction accuracy. The buffers are planar FP16.
//
// Fully async: stats are written to device buffers and the transfer kernel
// reads them directly from device memory. No cudaStreamSynchronize or
// cudaMemcpy to host needed.

#include "color_transfer.h"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Reduction kernel: sum and sum-of-squares per channel
// ---------------------------------------------------------------------------
// Each block reduces a chunk of pixels. Output: partial sums per block per channel.
// Layout: partialSums[blockIdx.x * 6 + 0..2] = channel sums
//         partialSums[blockIdx.x * 6 + 3..5] = channel sum-of-squares

static constexpr int REDUCE_BLOCK_SIZE = 256;

__global__ void kernel_channel_reduce(
    const __half* __restrict__ rgb,
    float* __restrict__ partialSums,
    int srcWidth, int srcHeight,
    int padWidth, int padHeight
) {
    // Shared memory: 6 values per thread (sum_r, sum_g, sum_b, sq_r, sq_g, sq_b)
    __shared__ float sSum[REDUCE_BLOCK_SIZE * 6];

    int tid = threadIdx.x;
    int planeSize = padWidth * padHeight;
    int nPixels = srcWidth * srcHeight;

    // Initialize shared memory
    for (int c = 0; c < 6; c++) {
        sSum[tid * 6 + c] = 0.0f;
    }

    // Each thread accumulates multiple pixels (grid-stride loop)
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int pixIdx = globalIdx; pixIdx < nPixels; pixIdx += stride) {
        int y = pixIdx / srcWidth;
        int x = pixIdx % srcWidth;
        int padIdx = y * padWidth + x;

        float r = __half2float(rgb[padIdx]);
        float g = __half2float(rgb[padIdx + planeSize]);
        float b = __half2float(rgb[padIdx + planeSize * 2]);

        sSum[tid * 6 + 0] += r;
        sSum[tid * 6 + 1] += g;
        sSum[tid * 6 + 2] += b;
        sSum[tid * 6 + 3] += r * r;
        sSum[tid * 6 + 4] += g * g;
        sSum[tid * 6 + 5] += b * b;
    }

    __syncthreads();

    // Tree reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            for (int c = 0; c < 6; c++) {
                sSum[tid * 6 + c] += sSum[(tid + s) * 6 + c];
            }
        }
        __syncthreads();
    }

    // Thread 0 writes block result
    if (tid == 0) {
        for (int c = 0; c < 6; c++) {
            partialSums[blockIdx.x * 6 + c] = sSum[c];
        }
    }
}

// ---------------------------------------------------------------------------
// Final reduction: sum partial block results -> mean and std
// ---------------------------------------------------------------------------
__global__ void kernel_finalize_stats(
    const float* __restrict__ partialSums,
    float* __restrict__ output,   // 6 floats: mean_r, mean_g, mean_b, std_r, std_g, std_b
    int nBlocks,
    int nPixels
) {
    // Single block, 3 threads (one per channel) -- trivial final reduction
    int c = threadIdx.x;
    if (c >= 3) return;

    float sum = 0.0f;
    float sumSq = 0.0f;
    for (int i = 0; i < nBlocks; i++) {
        sum   += partialSums[i * 6 + c];
        sumSq += partialSums[i * 6 + c + 3];
    }

    float mean = sum / (float)nPixels;
    float variance = sumSq / (float)nPixels - mean * mean;
    // Clamp variance to avoid sqrt of negative due to floating point
    float stddev = sqrtf(fmaxf(variance, 1e-8f));

    output[c]     = mean;
    output[c + 3] = stddev;
}

// ---------------------------------------------------------------------------
// Apply color transfer: reads stats from device memory, no host round-trip
// ---------------------------------------------------------------------------
__global__ void kernel_apply_color_transfer(
    __half* __restrict__ rgb,
    int srcWidth, int srcHeight,
    int padWidth, int padHeight,
    const float* __restrict__ inputStats,   // 6 floats: mean_r,g,b, std_r,g,b
    const float* __restrict__ outputStats   // 6 floats: mean_r,g,b, std_r,g,b
) {
    // Load stats into shared memory once per block (6 reads from global memory is
    // negligible, but shared memory avoids redundant global loads across warps)
    __shared__ float sInMean[3], sInStd[3], sOutMean[3], sOutStd[3];

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        sInMean[0]  = inputStats[0];
        sInMean[1]  = inputStats[1];
        sInMean[2]  = inputStats[2];
        sInStd[0]   = inputStats[3];
        sInStd[1]   = inputStats[4];
        sInStd[2]   = inputStats[5];
        sOutMean[0] = outputStats[0];
        sOutMean[1] = outputStats[1];
        sOutMean[2] = outputStats[2];
        sOutStd[0]  = outputStats[3];
        sOutStd[1]  = outputStats[4];
        sOutStd[2]  = outputStats[5];
    }
    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= srcWidth || y >= srcHeight) return;

    int planeSize = padWidth * padHeight;
    int idx = y * padWidth + x;

    // Pre-compute scale factors
    float scaleR = sInStd[0] / fmaxf(sOutStd[0], 1e-8f);
    float scaleG = sInStd[1] / fmaxf(sOutStd[1], 1e-8f);
    float scaleB = sInStd[2] / fmaxf(sOutStd[2], 1e-8f);

    float r = __half2float(rgb[idx]);
    float g = __half2float(rgb[idx + planeSize]);
    float b = __half2float(rgb[idx + planeSize * 2]);

    r = (r - sOutMean[0]) * scaleR + sInMean[0];
    g = (g - sOutMean[1]) * scaleG + sInMean[1];
    b = (b - sOutMean[2]) * scaleB + sInMean[2];

    // Clamp to [0, 1]
    r = fminf(fmaxf(r, 0.0f), 1.0f);
    g = fminf(fmaxf(g, 0.0f), 1.0f);
    b = fminf(fmaxf(b, 0.0f), 1.0f);

    rgb[idx]                 = __float2half(r);
    rgb[idx + planeSize]     = __float2half(g);
    rgb[idx + planeSize * 2] = __float2half(b);
}

// ---------------------------------------------------------------------------
// Host-callable wrappers
// ---------------------------------------------------------------------------

// Device workspace layout:
//   [0 .. nBlocks*6-1] : partial sums from reduction blocks
//   [nBlocks*6 .. nBlocks*6+5] : final 6 floats (mean_r,g,b, std_r,g,b)
//
// The deviceWorkspace must be allocated by the caller with enough space.
// Minimum size: (nBlocks + 1) * 6 * sizeof(float), where nBlocks = ceil(nPixels / REDUCE_BLOCK_SIZE).
// In practice, for 1080p: ceil(1920*1080/256) = 8100 blocks -> ~194KB. Negligible.

void launchComputeChannelStats(
    const __half* rgb,
    int srcWidth, int srcHeight,
    int padWidth, int padHeight,
    float* deviceStatsOut,
    float* deviceWorkspace,
    cudaStream_t stream
) {
    int nPixels = srcWidth * srcHeight;
    int nBlocks = (nPixels + REDUCE_BLOCK_SIZE - 1) / REDUCE_BLOCK_SIZE;
    // Cap blocks to avoid excessive partial sums (each pixel still covered via grid stride)
    if (nBlocks > 1024) nBlocks = 1024;

    float* partialSums = deviceWorkspace;
    float* finalStats  = deviceStatsOut;  // Write directly to caller's device buffer

    // Pass 1: parallel reduction into partial sums
    kernel_channel_reduce<<<nBlocks, REDUCE_BLOCK_SIZE, 0, stream>>>(
        rgb, partialSums, srcWidth, srcHeight, padWidth, padHeight);

    // Pass 2: finalize (single block, 3 threads) -- writes to deviceStatsOut
    kernel_finalize_stats<<<1, 3, 0, stream>>>(
        partialSums, finalStats, nBlocks, nPixels);

    // No sync, no memcpy to host -- stats stay on device for the transfer kernel
}

void launchApplyColorTransfer(
    __half* rgb,
    int srcWidth, int srcHeight,
    int padWidth, int padHeight,
    const float* inputStatsDevice,
    const float* outputStatsDevice,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((srcWidth + block.x - 1) / block.x,
              (srcHeight + block.y - 1) / block.y);

    kernel_apply_color_transfer<<<grid, block, 0, stream>>>(
        rgb, srcWidth, srcHeight, padWidth, padHeight,
        inputStatsDevice, outputStatsDevice);
}
