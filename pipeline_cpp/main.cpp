// main.cpp -- GPU-only video enhancement pipeline
//
// NVDEC decode -> CUDA color convert -> TensorRT inference -> CUDA color convert -> NVENC encode
//
// All frames stay on the GPU. No Python, no VapourSynth, no stdio pipes.
// Uses triple-buffered pipelining with CUDA streams for maximum throughput.

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <functional>
#include <chrono>
#include <string>
#include <cstring>
#include <cstdio>

// Video SDK (include paths set by CMake to match SDK's own include style)
#include "NvEncoder/NvEncoderCuda.h"
#include "NvDecoder/NvDecoder.h"
#include "NvCodecUtils.h"
#include "FFmpegDemuxer.h"

// Our modules
#include "trt_inference.h"
#include "color_kernels.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
struct PipelineConfig {
    std::string inputPath;
    std::string outputPath;
    std::string enginePath;
    int gpu          = 0;
    int cq           = 24;     // constant quality (lower = higher quality)
    std::string preset = "p4"; // NVENC preset
    bool tenBit      = false;  // output 10-bit HEVC
};

static void printUsage() {
    std::cerr
        << "Usage: remaster_pipeline [options]\n"
        << "\n"
        << "  --input   / -i   Input video file (required)\n"
        << "  --output  / -o   Output HEVC bitstream file (required)\n"
        << "  --engine  / -e   TensorRT engine file (required)\n"
        << "  --gpu            GPU ordinal (default: 0)\n"
        << "  --cq             Constant quality value, lower=better (default: 24)\n"
        << "  --preset         NVENC preset: p1..p7 (default: p4)\n"
        << "  --10bit          Output 10-bit HEVC\n"
        << "  --help   / -h    Show this message\n"
        << std::endl;
}

static bool parseArgs(int argc, char** argv, PipelineConfig& cfg) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage();
            return false;
        }
        else if ((arg == "--input" || arg == "-i") && i+1 < argc)   cfg.inputPath  = argv[++i];
        else if ((arg == "--output" || arg == "-o") && i+1 < argc)  cfg.outputPath = argv[++i];
        else if ((arg == "--engine" || arg == "-e") && i+1 < argc)  cfg.enginePath = argv[++i];
        else if (arg == "--gpu" && i+1 < argc)                      cfg.gpu = std::stoi(argv[++i]);
        else if (arg == "--cq" && i+1 < argc)                       cfg.cq  = std::stoi(argv[++i]);
        else if (arg == "--preset" && i+1 < argc)                   cfg.preset = argv[++i];
        else if (arg == "--10bit")                                   cfg.tenBit = true;
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage();
            return false;
        }
    }

    if (cfg.inputPath.empty() || cfg.outputPath.empty() || cfg.enginePath.empty()) {
        std::cerr << "Error: --input, --output, and --engine are required." << std::endl;
        printUsage();
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// NVENC preset string -> GUID
// ---------------------------------------------------------------------------
static GUID presetGuidFromString(const std::string& preset) {
    if (preset == "p1") return NV_ENC_PRESET_P1_GUID;
    if (preset == "p2") return NV_ENC_PRESET_P2_GUID;
    if (preset == "p3") return NV_ENC_PRESET_P3_GUID;
    if (preset == "p4") return NV_ENC_PRESET_P4_GUID;
    if (preset == "p5") return NV_ENC_PRESET_P5_GUID;
    if (preset == "p6") return NV_ENC_PRESET_P6_GUID;
    if (preset == "p7") return NV_ENC_PRESET_P7_GUID;
    std::cerr << "Unknown preset '" << preset << "', using p4" << std::endl;
    return NV_ENC_PRESET_P4_GUID;
}

// ---------------------------------------------------------------------------
// Frame buffer: holds GPU memory for one pipeline stage
// ---------------------------------------------------------------------------
struct FrameBuffer {
    __half* rgbIn  = nullptr;  // Input  to TRT: 3 * padH * padW half floats
    __half* rgbOut = nullptr;  // Output of TRT: same size
    size_t  rgbSize = 0;

    void allocate(int padW, int padH) {
        rgbSize = (size_t)3 * padW * padH * sizeof(__half);
        cudaMalloc(&rgbIn,  rgbSize);
        cudaMalloc(&rgbOut, rgbSize);
    }

    void free() {
        if (rgbIn)  { cudaFree(rgbIn);  rgbIn  = nullptr; }
        if (rgbOut) { cudaFree(rgbOut); rgbOut = nullptr; }
    }
};

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    PipelineConfig cfg;
    if (!parseArgs(argc, argv, cfg))
        return 1;

    try {

    // -----------------------------------------------------------------------
    // 1. Initialize CUDA
    // -----------------------------------------------------------------------
    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (cfg.gpu < 0 || cfg.gpu >= nGpu) {
        std::cerr << "GPU " << cfg.gpu << " out of range [0, " << nGpu-1 << "]" << std::endl;
        return 1;
    }

    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, cfg.gpu));
    char deviceName[128];
    ck(cuDeviceGetName(deviceName, sizeof(deviceName), cuDevice));
    std::cerr << "GPU: " << deviceName << std::endl;

    CUcontext cuContext = nullptr;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    // -----------------------------------------------------------------------
    // 2. Open input with FFmpeg demuxer + NVDEC decoder
    // -----------------------------------------------------------------------
    FFmpegDemuxer demuxer(cfg.inputPath.c_str());
    int srcWidth  = demuxer.GetWidth();
    int srcHeight = demuxer.GetHeight();
    int bitDepth  = demuxer.GetBitDepth();
    bool is10bit  = bitDepth > 8;

    std::cerr << "Input: " << srcWidth << "x" << srcHeight
              << " " << bitDepth << "-bit "
              << (demuxer.GetVideoCodec() == AV_CODEC_ID_HEVC ? "HEVC" : "H264")
              << std::endl;

    NvDecoder dec(cuContext, srcWidth, srcHeight, true,
                  FFmpeg2NvCodecId(demuxer.GetVideoCodec()),
                  nullptr, false, true);

    // -----------------------------------------------------------------------
    // 3. Load TensorRT engine
    // -----------------------------------------------------------------------
    TrtInference trt;
    if (!trt.loadEngine(cfg.enginePath)) {
        std::cerr << "Failed to load TRT engine" << std::endl;
        return 1;
    }

    // Padded dimensions (model input size, must be divisible by 8)
    int padW = alignTo8(srcWidth);
    int padH = alignTo8(srcHeight);

    // Verify engine matches
    if (trt.getInputWidth() != padW || trt.getInputHeight() != padH) {
        std::cerr << "Warning: Engine expects " << trt.getInputWidth() << "x" << trt.getInputHeight()
                  << " but video is " << srcWidth << "x" << srcHeight
                  << " (padded: " << padW << "x" << padH << ")" << std::endl;
        // Use engine dimensions if they differ -- the engine is authoritative
        padW = trt.getInputWidth();
        padH = trt.getInputHeight();
    }

    // -----------------------------------------------------------------------
    // 4. Initialize NVENC encoder
    // -----------------------------------------------------------------------
    bool outIs10bit = cfg.tenBit;
    NV_ENC_BUFFER_FORMAT encFormat = outIs10bit
        ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT
        : NV_ENC_BUFFER_FORMAT_NV12;

    auto encDeleteFunc = [](NvEncoderCuda* pEnc) {
        if (pEnc) { pEnc->DestroyEncoder(); delete pEnc; }
    };
    using NvEncCudaPtr = std::unique_ptr<NvEncoderCuda, decltype(encDeleteFunc)>;
    NvEncCudaPtr pEnc(new NvEncoderCuda(cuContext, srcWidth, srcHeight, encFormat),
                      encDeleteFunc);

    {
        NV_ENC_INITIALIZE_PARAMS initParams = { NV_ENC_INITIALIZE_PARAMS_VER };
        NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
        initParams.encodeConfig = &encodeConfig;

        GUID codecGuid  = NV_ENC_CODEC_HEVC_GUID;
        GUID presetGuid = presetGuidFromString(cfg.preset);

        pEnc->CreateDefaultEncoderParams(&initParams, codecGuid, presetGuid);

        // Apply tuning
        initParams.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;

        // VBR with constant quality
        encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;
        encodeConfig.rcParams.constQP = { (uint32_t)cfg.cq, (uint32_t)cfg.cq, (uint32_t)cfg.cq };
        encodeConfig.rcParams.multiPass = NV_ENC_TWO_PASS_FULL_RESOLUTION;

        // HEVC-specific
        if (outIs10bit) {
            encodeConfig.encodeCodecConfig.hevcConfig.pixelBitDepthMinus8 = 2;
            encodeConfig.encodeCodecConfig.hevcConfig.inputBitDepthMinus8 = 2;
        }

        // B-frames for quality
        encodeConfig.frameIntervalP = 3; // 2 B-frames

        pEnc->CreateEncoder(&initParams);
    }

    std::cerr << "Encoder: HEVC " << cfg.preset << " cq=" << cfg.cq
              << (outIs10bit ? " 10-bit" : " 8-bit") << std::endl;

    // -----------------------------------------------------------------------
    // 5. Allocate pipeline buffers and CUDA streams
    // -----------------------------------------------------------------------
    // Triple-buffer for decode/infer/encode overlap
    constexpr int NUM_BUFFERS = 3;
    FrameBuffer buffers[NUM_BUFFERS];
    cudaStream_t streams[NUM_BUFFERS];
    cudaEvent_t  events[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i].allocate(padW, padH);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    // Open output file
    std::ofstream fpOut(cfg.outputPath, std::ios::out | std::ios::binary);
    if (!fpOut) {
        std::cerr << "Failed to open output file: " << cfg.outputPath << std::endl;
        return 1;
    }

    // -----------------------------------------------------------------------
    // 6. Pipeline loop
    // -----------------------------------------------------------------------
    auto startTime = std::chrono::high_resolution_clock::now();
    int nFrameDecoded = 0;
    int nFrameEncoded = 0;
    int nVideoBytes = 0;
    uint8_t* pVideo = nullptr;
    uint8_t** ppFrame = nullptr;
    int nFrameReturned = 0;

    // Simple sequential pipeline for correctness first.
    // Each frame: demux -> decode -> color convert -> infer -> color convert -> encode -> write
    // We still use async CUDA operations within each frame for GPU pipelining.
    cudaStream_t inferStream = streams[0];

    auto reportProgress = [&]() {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        double fps = (elapsed > 0) ? nFrameEncoded / elapsed : 0.0;
        fprintf(stderr, "\rFrame %d | %.1f fps | %.1fs elapsed",
                nFrameEncoded, fps, elapsed);
    };

    // Lambda: process decoded frames through inference and encoding
    auto processDecodedFrames = [&](uint8_t** ppFrame, int nFrameReturned) {
        for (int i = 0; i < nFrameReturned; i++) {
            nFrameDecoded++;
            FrameBuffer& buf = buffers[nFrameDecoded % NUM_BUFFERS];

            // -- Color convert: NV12/P010 -> planar RGB FP16 --
            int decPitch = dec.GetDeviceFramePitch();
            if (is10bit) {
                launchP010ToRgbFp16(
                    ppFrame[i], buf.rgbIn,
                    srcWidth, srcHeight, decPitch,
                    padW, padH,
                    inferStream);
            } else {
                launchNv12ToRgbFp16(
                    ppFrame[i], buf.rgbIn,
                    srcWidth, srcHeight, decPitch,
                    padW, padH,
                    inferStream);
            }

            // -- TRT inference --
            if (!trt.infer(buf.rgbIn, buf.rgbOut, inferStream)) {
                std::cerr << "\nInference failed at frame " << nFrameDecoded << std::endl;
                return;
            }

            // -- Color convert: planar RGB FP16 -> NV12/P010 for encoder --
            const NvEncInputFrame* encFrame = pEnc->GetNextInputFrame();

            if (outIs10bit) {
                launchRgbFp16ToP010(
                    buf.rgbOut, (uint8_t*)encFrame->inputPtr,
                    srcWidth, srcHeight,
                    padW, padH,
                    encFrame->pitch,
                    inferStream);
            } else {
                launchRgbFp16ToNv12(
                    buf.rgbOut, (uint8_t*)encFrame->inputPtr,
                    srcWidth, srcHeight,
                    padW, padH,
                    encFrame->pitch,
                    inferStream);
            }

            // Sync before encode (NVENC needs the data ready)
            cudaStreamSynchronize(inferStream);

            // -- Encode --
            std::vector<std::vector<uint8_t>> vPacket;
            pEnc->EncodeFrame(vPacket);

            for (auto& packet : vPacket) {
                fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
                nFrameEncoded++;
            }

            // Progress
            if (nFrameDecoded % 30 == 0) {
                reportProgress();
            }
        }
    };

    // Demux and decode loop
    while (demuxer.Demux(&pVideo, &nVideoBytes)) {
        dec.Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
        processDecodedFrames(ppFrame, nFrameReturned);
    }

    // Flush decoder: send empty packet to get buffered frames
    dec.Decode(nullptr, 0, &ppFrame, &nFrameReturned);
    processDecodedFrames(ppFrame, nFrameReturned);

    // -----------------------------------------------------------------------
    // 7. Flush encoder
    // -----------------------------------------------------------------------
    {
        std::vector<std::vector<uint8_t>> vPacket;
        pEnc->EndEncode(vPacket);
        for (auto& packet : vPacket) {
            fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
            nFrameEncoded++;
        }
    }

    // -----------------------------------------------------------------------
    // 8. Report results and clean up
    // -----------------------------------------------------------------------
    fpOut.close();

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();
    double avgFps = nFrameEncoded / totalTime;

    fprintf(stderr, "\n\nDone: %d frames in %.2f seconds (%.1f fps avg)\n",
            nFrameEncoded, totalTime, avgFps);
    fprintf(stderr, "Output: %s\n", cfg.outputPath.c_str());

    // Free pipeline buffers
    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i].free();
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    // Encoder is cleaned up by unique_ptr
    // Decoder destructor handles cleanup

    return 0;

    } catch (const std::exception& ex) {
        std::cerr << "\nFatal error: " << ex.what() << std::endl;
        return 1;
    }
}
