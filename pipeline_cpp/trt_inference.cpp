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

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputName_ = name;
            // Expect NCHW: [1, 3, H, W]
            if (dims.nbDims >= 4) {
                inputH_ = dims.d[2];
                inputW_ = dims.d[3];
            }
            std::cerr << "Input tensor '" << name << "': "
                      << dims.d[0] << "x" << dims.d[1] << "x"
                      << dims.d[2] << "x" << dims.d[3] << std::endl;
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
            outputName_ = name;
            if (dims.nbDims >= 4) {
                outputH_ = dims.d[2];
                outputW_ = dims.d[3];
            }
            std::cerr << "Output tensor '" << name << "': "
                      << dims.d[0] << "x" << dims.d[1] << "x"
                      << dims.d[2] << "x" << dims.d[3] << std::endl;
        }
    }

    if (inputName_.empty() || outputName_.empty()) {
        std::cerr << "Could not find input/output tensors" << std::endl;
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
