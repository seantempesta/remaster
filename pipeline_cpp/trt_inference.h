// trt_inference.h -- TensorRT engine loader and inference wrapper
// Updated for TensorRT 10.16 API (tensor-name-based, enqueueV3)
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
    enum class DataType : int32_t;
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

    // Run inference via CUDA graph for reduced launch overhead.
    // bufferIdx selects which graph slot (0..kMaxGraphs-1) -- use a different
    // slot for each distinct (input, output) pointer pair.
    // Falls back to infer() if graph capture fails or bufferIdx is out of range.
    bool inferWithGraph(__half* input, __half* output, cudaStream_t stream, int bufferIdx);

    // Query engine dimensions
    int getInputHeight() const { return inputH_; }
    int getInputWidth() const { return inputW_; }
    int getOutputHeight() const { return outputH_; }
    int getOutputWidth() const { return outputW_; }

private:
    // TRT objects (opaque pointers, cleaned up in destructor via delete)
    nvinfer1::IRuntime*          runtime_ = nullptr;
    nvinfer1::ICudaEngine*       engine_  = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    nvinfer1::ILogger*           logger_  = nullptr;

    int inputH_  = 0, inputW_  = 0;
    int outputH_ = 0, outputW_ = 0;

    // Tensor names (TRT 10.x uses names instead of binding indices)
    std::string inputName_;
    std::string outputName_;

    // I/O data types (validated to be FP16 at load time)
    nvinfer1::DataType inputDtype_;
    nvinfer1::DataType outputDtype_;

    // CUDA graph capture state
    static constexpr int kMaxGraphs = 4;
    cudaGraph_t       graphs_[kMaxGraphs] = {};
    cudaGraphExec_t   graphExecs_[kMaxGraphs] = {};
    bool              graphCaptured_[kMaxGraphs] = {};
    bool              graphSupported_ = true;  // set false if capture fails
};
