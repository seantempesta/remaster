// trt_inference.cpp -- TensorRT engine loader and inference wrapper
// Updated for TensorRT 10.6+ API (no .destroy(), tensor-name-based API, enqueueV3)

#include "trt_inference.h"
#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <vector>

// ---------------------------------------------------------------------------
// TRT logger that routes to stderr
// ---------------------------------------------------------------------------
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only print warnings and above
        if (severity <= Severity::kWARNING) {
            const char* level = "?";
            switch (severity) {
                case Severity::kINTERNAL_ERROR: level = "INTERNAL_ERROR"; break;
                case Severity::kERROR:          level = "ERROR"; break;
                case Severity::kWARNING:        level = "WARNING"; break;
                default: break;
            }
            std::cerr << "[TRT " << level << "] " << msg << std::endl;
        }
    }
};

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

TrtInference::~TrtInference() {
    // Clean up CUDA graphs
    for (int i = 0; i < kMaxGraphs; i++) {
        if (graphExecs_[i]) cudaGraphExecDestroy(graphExecs_[i]);
        if (graphs_[i]) cudaGraphDestroy(graphs_[i]);
    }
    // TRT 10.x uses normal C++ destructors, not .destroy()
    delete context_;
    context_ = nullptr;
    delete engine_;
    engine_ = nullptr;
    delete runtime_;
    runtime_ = nullptr;
    delete logger_;
    logger_ = nullptr;
}

bool TrtInference::loadEngine(const std::string& enginePath) {
    // Read engine file
    std::ifstream file(enginePath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open engine file: " << enginePath << std::endl;
        return false;
    }

    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engineData(fileSize);
    if (!file.read(engineData.data(), fileSize)) {
        std::cerr << "Failed to read engine file" << std::endl;
        return false;
    }
    file.close();

    std::cerr << "Loaded engine file: " << enginePath
              << " (" << (fileSize / 1024 / 1024) << " MB)" << std::endl;

    // Create runtime and engine
    logger_ = new TrtLogger();
    runtime_ = nvinfer1::createInferRuntime(*logger_);
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engineData.data(), fileSize);
    if (!engine_) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // Find input/output tensors using TRT 10.x tensor-name API
    int nbIO = engine_->getNbIOTensors();
    for (int i = 0; i < nbIO; i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);

        nvinfer1::DataType dtype = engine_->getTensorDataType(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputName_ = name;
            inputDtype_ = dtype;
            // Expect NCHW: [1, 3, H, W]
            if (dims.nbDims >= 4) {
                inputH_ = dims.d[2];
                inputW_ = dims.d[3];
            }
            std::cerr << "Input tensor '" << name << "': "
                      << dims.d[0] << "x" << dims.d[1] << "x"
                      << dims.d[2] << "x" << dims.d[3]
                      << " dtype=" << static_cast<int>(dtype) << std::endl;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName_ = name;
            outputDtype_ = dtype;
            if (dims.nbDims >= 4) {
                outputH_ = dims.d[2];
                outputW_ = dims.d[3];
            }
            std::cerr << "Output tensor '" << name << "': "
                      << dims.d[0] << "x" << dims.d[1] << "x"
                      << dims.d[2] << "x" << dims.d[3]
                      << " dtype=" << static_cast<int>(dtype) << std::endl;
        }
    }

    if (inputName_.empty() || outputName_.empty()) {
        std::cerr << "Could not find input/output tensors" << std::endl;
        return false;
    }

    // Validate I/O formats -- pipeline assumes FP16 buffers
    if (inputDtype_ != nvinfer1::DataType::kHALF || outputDtype_ != nvinfer1::DataType::kHALF) {
        auto dtypeName = [](nvinfer1::DataType d) -> const char* {
            switch (d) {
                case nvinfer1::DataType::kFLOAT: return "FP32";
                case nvinfer1::DataType::kHALF:  return "FP16";
                case nvinfer1::DataType::kINT8:  return "INT8";
                default: return "unknown";
            }
        };
        std::cerr << "ERROR: Engine has " << dtypeName(inputDtype_) << " input / "
                  << dtypeName(outputDtype_) << " output, but pipeline requires FP16 I/O.\n"
                  << "Rebuild engine from FP16 ONNX model, or add "
                     "--inputIOFormats=fp16:chw --outputIOFormats=fp16:chw" << std::endl;
        return false;
    }

    std::cerr << "TensorRT engine ready: " << inputW_ << "x" << inputH_
              << " -> " << outputW_ << "x" << outputH_ << std::endl;
    return true;
}

bool TrtInference::infer(__half* input, __half* output, cudaStream_t stream) {
    // TRT 10.x: set tensor addresses by name, then enqueueV3
    if (!context_->setTensorAddress(inputName_.c_str(), input)) {
        std::cerr << "Failed to set input tensor address" << std::endl;
        return false;
    }
    if (!context_->setTensorAddress(outputName_.c_str(), output)) {
        std::cerr << "Failed to set output tensor address" << std::endl;
        return false;
    }

    bool ok = context_->enqueueV3(stream);
    if (!ok) {
        std::cerr << "TensorRT enqueueV3 failed" << std::endl;
    }
    return ok;
}

bool TrtInference::inferWithGraph(__half* input, __half* output, cudaStream_t stream, int bufferIdx) {
    if (bufferIdx < 0 || bufferIdx >= kMaxGraphs) {
        return infer(input, output, stream);
    }

    if (!graphSupported_) {
        return infer(input, output, stream);
    }

    if (!graphCaptured_[bufferIdx]) {
        // Set tensor addresses (stable per bufferIdx)
        if (!context_->setTensorAddress(inputName_.c_str(), input)) return false;
        if (!context_->setTensorAddress(outputName_.c_str(), output)) return false;

        // Capture
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            std::cerr << "CUDA graph capture begin failed, falling back to regular inference" << std::endl;
            graphSupported_ = false;
            return infer(input, output, stream);
        }

        context_->enqueueV3(stream);

        err = cudaStreamEndCapture(stream, &graphs_[bufferIdx]);
        if (err != cudaSuccess || graphs_[bufferIdx] == nullptr) {
            std::cerr << "CUDA graph capture failed, falling back to regular inference" << std::endl;
            graphSupported_ = false;
            // Stream may be in capture error state, try to end capture properly
            cudaGraph_t dummyGraph = nullptr;
            cudaStreamEndCapture(stream, &dummyGraph);
            return infer(input, output, stream);
        }

        err = cudaGraphInstantiate(&graphExecs_[bufferIdx], graphs_[bufferIdx], 0);
        if (err != cudaSuccess) {
            std::cerr << "CUDA graph instantiation failed, falling back" << std::endl;
            graphSupported_ = false;
            cudaGraphDestroy(graphs_[bufferIdx]);
            graphs_[bufferIdx] = nullptr;
            return infer(input, output, stream);
        }

        graphCaptured_[bufferIdx] = true;
        std::cerr << "CUDA graph captured for buffer " << bufferIdx << std::endl;
    }

    // Launch captured graph
    cudaError_t err = cudaGraphLaunch(graphExecs_[bufferIdx], stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA graph launch failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}
