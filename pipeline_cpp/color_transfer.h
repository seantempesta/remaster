// color_transfer.h -- Per-frame color/brightness transfer for HDR/BT.2020 content
//
// Computes per-channel mean and std of the input RGB buffer before inference,
// then transfers those statistics onto the model output after inference.
// This corrects color shifts when BT.2020 content is processed through BT.709
// color kernels, without requiring full BT.2020 matrix support.
//
// All operations are fully async on the GPU -- no host synchronization needed.
// Stats are written to device buffers and read directly by the transfer kernel.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Compute per-channel mean and standard deviation of a planar RGB FP16 buffer.
// Buffer layout: [R plane: H*W] [G plane: H*W] [B plane: H*W]
// Only pixels within [srcWidth x srcHeight] are considered (ignoring padding).
//
// Results are written to deviceStatsOut (6 floats on device):
//   [mean_r, mean_g, mean_b, std_r, std_g, std_b]
//
// No host synchronization -- all work stays on the GPU stream.
void launchComputeChannelStats(
    const __half* rgb,
    int srcWidth, int srcHeight,
    int padWidth, int padHeight,
    float* deviceStatsOut,          // 6 floats output on device
    float* deviceWorkspace,         // workspace for partial sums
    cudaStream_t stream);

// Apply color transfer in-place: adjusts output to match input's color statistics.
//   output[c][y][x] = (output[c][y][x] - outMean[c]) * (inStd[c] / outStd[c]) + inMean[c]
// Clamped to [0, 1]. Operates on padded buffer but only modifies [srcWidth x srcHeight].
//
// inputStatsDevice and outputStatsDevice are device pointers to 6 floats each
// (as written by launchComputeChannelStats). No host copy needed.
void launchApplyColorTransfer(
    __half* rgb,
    int srcWidth, int srcHeight,
    int padWidth, int padHeight,
    const float* inputStatsDevice,  // 6 floats: input mean_r,g,b + std_r,g,b
    const float* outputStatsDevice, // 6 floats: output mean_r,g,b + std_r,g,b
    cudaStream_t stream);
