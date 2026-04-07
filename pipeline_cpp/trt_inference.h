// trt_inference.h -- TensorRT engine loader and inference wrapper
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Forward-declare TRT types to avoid pulling NvInfer.h into every TU
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
    class ILogger;
}

class TrtInference {
public:
    TrtInference() = default;
    ~TrtInference();

    // Non-copyable
    TrtInference(const TrtInference&) = delete;
    TrtInference& operator=(const TrtInference&) = delete;

    // Load a pre-built TensorRT engine from file.
    // Returns false on failure (logs errors to stderr).
    bool loadEngine(const std::string& enginePath);

    // Run inference. Input and output are device pointers to FP16 planar RGB
    // with shape [1, 3, height, width]. Async on the given stream.
    bool infer(__half* input, __half* output, cudaStream_t stream);

    // Query engine dimensions
    int getInputHeight() const { return inputH_; }
    int getInputWidth() const { return inputW_; }
    int getOutputHeight() const { return outputH_; }
    int getOutputWidth() const { return outputW_; }

private:
    // TRT objects (opaque pointers, cleaned up in destructor)
    nvinfer1::IRuntime*          runtime_ = nullptr;
    nvinfer1::ICudaEngine*       engine_  = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    nvinfer1::ILogger*           logger_  = nullptr;

    int inputH_  = 0, inputW_  = 0;
    int outputH_ = 0, outputW_ = 0;

    // Binding indices
    int inputIndex_  = -1;
    int outputIndex_ = -1;
};
