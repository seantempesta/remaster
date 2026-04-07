// trt_inference.cpp -- TensorRT engine loader and inference wrapper

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
    if (context_) { context_->destroy(); context_ = nullptr; }
    if (engine_)  { engine_->destroy();  engine_  = nullptr; }
    if (runtime_) { runtime_->destroy(); runtime_ = nullptr; }
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

    // Find input/output bindings
    int nbBindings = engine_->getNbBindings();
    for (int i = 0; i < nbBindings; i++) {
        const char* name = engine_->getBindingName(i);
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);

        if (engine_->bindingIsInput(i)) {
            inputIndex_ = i;
            // Expect NCHW: [1, 3, H, W]
            if (dims.nbDims >= 4) {
                inputH_ = dims.d[2];
                inputW_ = dims.d[3];
            }
            std::cerr << "Input binding '" << name << "': "
                      << dims.d[0] << "x" << dims.d[1] << "x"
                      << dims.d[2] << "x" << dims.d[3] << std::endl;
        } else {
            outputIndex_ = i;
            if (dims.nbDims >= 4) {
                outputH_ = dims.d[2];
                outputW_ = dims.d[3];
            }
            std::cerr << "Output binding '" << name << "': "
                      << dims.d[0] << "x" << dims.d[1] << "x"
                      << dims.d[2] << "x" << dims.d[3] << std::endl;
        }
    }

    if (inputIndex_ < 0 || outputIndex_ < 0) {
        std::cerr << "Could not find input/output bindings" << std::endl;
        return false;
    }

    std::cerr << "TensorRT engine ready: " << inputW_ << "x" << inputH_
              << " -> " << outputW_ << "x" << outputH_ << std::endl;
    return true;
}

bool TrtInference::infer(__half* input, __half* output, cudaStream_t stream) {
    void* bindings[2];
    bindings[inputIndex_]  = input;
    bindings[outputIndex_] = output;

    bool ok = context_->enqueueV2(bindings, stream, nullptr);
    if (!ok) {
        std::cerr << "TensorRT enqueueV2 failed" << std::endl;
    }
    return ok;
}
