// main.cpp -- GPU video enhancement pipeline
//
// Decode -> CUDA color convert -> TensorRT inference -> CUDA color convert -> NVENC encode
//
// Decoding: NVDEC hardware by default, automatic fallback to FFmpeg software decode
// for codecs/profiles NVDEC doesn't support (e.g., H264 High 10-bit on RTX 3060).
// Use --sw-decode to force software decode.
//
// All processed frames stay on the GPU. No Python, no VapourSynth, no stdio pipes.
// Output is a proper MKV container with audio/subtitle passthrough.

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

// Video SDK
#include "NvEncoderCuda.h"          // local copy with SDK 13.0 nvEncodeAPI.h
#include "NvDecoder/NvDecoder.h"    // from submodule (SDK 8.1 NVDEC still works)
#include "NvCodecUtils.h"

// Our modules
#include "simple_demuxer.h"
#include "sw_decoder.h"
#include "trt_inference.h"
#include "color_kernels.h"
#include "color_transfer.h"
#include "mkv_muxer.h"
#include "async_writer.h"

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
    std::string preset = "p4"; // NVENC preset (SDK 10+: p1-p7, p1=fastest, p7=best quality)
    bool tenBit      = true;   // 10-bit HEVC output (matches source quality)
    bool noAudio     = false;  // skip audio/subtitle passthrough
    bool swDecode    = false;  // force FFmpeg software decode (skip NVDEC)
    bool noColorXfer = false;  // disable per-frame color/brightness transfer
};

static void printUsage() {
    std::cerr
        << "Usage: remaster_pipeline [options]\n"
        << "\n"
        << "  --input   / -i   Input video file (required)\n"
        << "  --output  / -o   Output MKV file (required)\n"
        << "  --engine  / -e   TensorRT engine file (required)\n"
        << "  --gpu            GPU ordinal (default: 0)\n"
        << "  --cq             Constant quality value, lower=better (default: 24)\n"
        << "  --preset         NVENC preset: p1-p7 (p1=fastest, p7=best quality, default: p4)\n"
        << "  --10bit          Output 10-bit HEVC\n"
        << "  --no-audio       Skip audio/subtitle passthrough\n"
        << "  --sw-decode      Force FFmpeg software decode (skip NVDEC)\n"
        << "  --no-color-transfer  Disable per-frame color/brightness matching\n"
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
        else if (arg == "--no-audio")                                cfg.noAudio = true;
        else if (arg == "--sw-decode")                              cfg.swDecode = true;
        else if (arg == "--no-color-transfer")                     cfg.noColorXfer = true;
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
// NVENC preset string -> GUID (SDK 10+ P1-P7 presets)
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
// NVENC tuning info from preset string
// ---------------------------------------------------------------------------
static NV_ENC_TUNING_INFO tuningFromPreset(const std::string& /*preset*/) {
    return NV_ENC_TUNING_INFO_HIGH_QUALITY;
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
    CUctxCreateParams ctxParams = {};  // CUDA 13.x v4 API (NULL params = regular context)
    ck(cuCtxCreate(&cuContext, &ctxParams, 0, cuDevice));

    // Set persistent L2 cache to 50% for TRT inference benefit
    {
        int l2CacheSize = 0;
        cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, cfg.gpu);
        if (l2CacheSize > 0) {
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2CacheSize / 2);
            std::cerr << "L2 persistent cache: " << (l2CacheSize / 2 / 1024) << " KB" << std::endl;
        }
    }

    // -----------------------------------------------------------------------
    // 2. Open input with SimpleDemuxer + decoder (NVDEC or software fallback)
    // -----------------------------------------------------------------------
    SimpleDemuxer demuxer(cfg.inputPath.c_str());
    if (!demuxer.IsValid()) {
        std::cerr << "Failed to open input: " << cfg.inputPath << std::endl;
        return 1;
    }

    int srcWidth  = demuxer.GetWidth();
    int srcHeight = demuxer.GetHeight();
    int bitDepth  = demuxer.GetBitDepth();
    bool is10bit  = bitDepth > 8;
    double fps    = demuxer.GetFrameRate();
    bool passAudio = !cfg.noAudio;

    if (fps <= 0.0) {
        std::cerr << "Warning: could not determine frame rate, assuming 23.976" << std::endl;
        fps = 24000.0 / 1001.0;
    }

    // Get codec name for display
    const char* codecName = "unknown";
    switch (demuxer.GetVideoCodec()) {
        case AV_CODEC_ID_HEVC: codecName = "HEVC"; break;
        case AV_CODEC_ID_H264: codecName = "H264"; break;
        case AV_CODEC_ID_VP9:  codecName = "VP9";  break;
        case AV_CODEC_ID_VP8:  codecName = "VP8";  break;
        case AV_CODEC_ID_AV1:  codecName = "AV1";  break;
        case AV_CODEC_ID_MPEG2VIDEO: codecName = "MPEG2"; break;
        case AV_CODEC_ID_MPEG4: codecName = "MPEG4"; break;
        default: {
            const AVCodecDescriptor* desc = avcodec_descriptor_get(demuxer.GetVideoCodec());
            if (desc) codecName = desc->name;
            break;
        }
    }

    std::cerr << "Input: " << srcWidth << "x" << srcHeight
              << " " << bitDepth << "-bit " << codecName
              << " @ " << fps << " fps"
              << std::endl;

    // Count audio/subtitle streams for info
    int nAudioStreams = 0, nSubStreams = 0;
    for (int i = 0; i < demuxer.GetNumStreams(); i++) {
        AVStream* s = demuxer.GetStream(i);
        if (s->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) nAudioStreams++;
        else if (s->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE) nSubStreams++;
    }
    if (passAudio) {
        std::cerr << "Passthrough: " << nAudioStreams << " audio, "
                  << nSubStreams << " subtitle stream(s)" << std::endl;
    }

    // Try NVDEC first, fall back to software decode if unsupported.
    // The --sw-decode flag forces software decode (useful for testing/debugging).
    bool useSwDecoder = cfg.swDecode;
    std::unique_ptr<NvDecoder> nvDec;
    std::unique_ptr<SwDecoder> swDec;

    if (!useSwDecoder) {
        // Check if NVDEC supports this codec + bit depth combination
        cudaVideoCodec nvCodec = FFmpeg2NvCodecId(demuxer.GetVideoCodec());
        if (nvCodec == cudaVideoCodec_NumCodecs) {
            // FFmpeg codec has no NVDEC equivalent (e.g., AV1 on older GPUs, obscure codecs)
            fprintf(stderr, "Decoder: %s has no NVDEC mapping, using software decode\n", codecName);
            useSwDecoder = true;
        } else {
            // Proactively check NVDEC capabilities before construction
            CUVIDDECODECAPS caps = {};
            caps.eCodecType = nvCodec;
            caps.eChromaFormat = cudaVideoChromaFormat_420;
            caps.nBitDepthMinus8 = (bitDepth > 8) ? (bitDepth - 8) : 0;

            CUresult capResult = cuvidGetDecoderCaps(&caps);
            if (capResult != CUDA_SUCCESS || !caps.bIsSupported) {
                fprintf(stderr, "Decoder: NVDEC does not support %s %d-bit on this GPU, "
                        "using software decode\n", codecName, bitDepth);
                useSwDecoder = true;
            } else {
                // NVDEC reports support, try to construct the decoder
                try {
                    nvDec = std::make_unique<NvDecoder>(
                        cuContext, srcWidth, srcHeight, true,
                        nvCodec, nullptr, false, true);
                    fprintf(stderr, "Decoder: NVDEC hardware\n");
                } catch (const NVDECException& ex) {
                    fprintf(stderr, "Decoder: NVDEC init failed (%s), using software decode\n",
                            ex.what());
                    useSwDecoder = true;
                }
            }
        }
    }

    if (useSwDecoder) {
        AVStream* videoStream = demuxer.GetStream(demuxer.GetVideoStreamIndex());
        swDec = std::make_unique<SwDecoder>(videoStream->codecpar);
        if (!swDec->IsValid()) {
            std::cerr << "Failed to initialize software decoder" << std::endl;
            return 1;
        }
        // Software decoder determines actual bit depth from the codec
        is10bit = swDec->Is10Bit();
        if (cfg.swDecode) {
            fprintf(stderr, "Decoder: FFmpeg software (forced via --sw-decode)\n");
        }
    }

    // Unified accessors for decoder pitch
    auto getDecoderPitch = [&]() -> int {
        if (nvDec) return nvDec->GetDeviceFramePitch();
        return swDec->GetDeviceFramePitch();
    };

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
        NV_ENC_TUNING_INFO tuning = tuningFromPreset(cfg.preset);

        pEnc->CreateDefaultEncoderParams(&initParams, codecGuid, presetGuid, tuning);

        // Rate control: Constant QP
        encodeConfig.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;
        encodeConfig.rcParams.constQP = { (uint32_t)cfg.cq, (uint32_t)cfg.cq, (uint32_t)cfg.cq };

        // GOP structure: keyframe every ~5 seconds for fast seeking
        int frameIntervalP = 3; // I-B-B-P pattern (2 B-frames for quality)
        uint32_t gopFrames = (uint32_t)(fps * 5.0 + 0.5);
        gopFrames = (gopFrames / frameIntervalP) * frameIntervalP;
        if (gopFrames < (uint32_t)frameIntervalP) gopFrames = (uint32_t)frameIntervalP;

        encodeConfig.gopLength = gopFrames;
        encodeConfig.frameIntervalP = frameIntervalP;

        // HEVC-specific settings
        auto& hevc = encodeConfig.encodeCodecConfig.hevcConfig;

        if (outIs10bit) {
            hevc.inputBitDepth  = NV_ENC_BIT_DEPTH_10;
            hevc.outputBitDepth = NV_ENC_BIT_DEPTH_10;
        }

        // IDR at every GOP boundary + repeat headers (required for seeking)
        hevc.idrPeriod = gopFrames;
        hevc.repeatSPSPPS = 1;

        // BT.709 color metadata (HD content standard)
        auto& vui = hevc.hevcVUIParameters;
        vui.videoSignalTypePresentFlag   = 1;
        vui.videoFormat                  = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
        vui.videoFullRangeFlag           = 0;  // Limited range
        vui.colourDescriptionPresentFlag = 1;
        vui.colourPrimaries              = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
        vui.transferCharacteristics      = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_BT709;
        vui.colourMatrix                 = NV_ENC_VUI_MATRIX_COEFFS_BT709;

        pEnc->CreateEncoder(&initParams);
    }

    std::cerr << "Encoder: HEVC " << cfg.preset << " cq=" << cfg.cq
              << (outIs10bit ? " 10-bit" : " 8-bit") << std::endl;

    // -----------------------------------------------------------------------
    // 5. Initialize MKV muxer + async writer
    // -----------------------------------------------------------------------
    MkvMuxer muxer;
    if (!muxer.open(demuxer.GetFormatContext(), demuxer.GetVideoStreamIndex(),
                    cfg.outputPath.c_str(),
                    srcWidth, srcHeight, fps,
                    outIs10bit, passAudio))
    {
        std::cerr << "Failed to initialize MKV muxer" << std::endl;
        return 1;
    }

    AsyncWriter writer(muxer);
    writer.start();

    std::cerr << "Output: " << cfg.outputPath << " (MKV)" << std::endl;

    // -----------------------------------------------------------------------
    // 6. Allocate pipeline buffers and CUDA streams
    // -----------------------------------------------------------------------
    constexpr int NUM_BUFFERS = 3;
    FrameBuffer buffers[NUM_BUFFERS];
    cudaStream_t streams[NUM_BUFFERS];
    cudaEvent_t  events[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i].allocate(padW, padH);
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    // -----------------------------------------------------------------------
    // 6b. Color transfer workspace (fully async, no host sync)
    // -----------------------------------------------------------------------
    // Workspace: 1024 blocks * 6 partial sums = 6144 floats for reduction
    // Stats buffers: 2 * 6 floats on device (input stats + output stats)
    // All operations stay on the GPU -- transfer kernel reads stats from device memory.
    float* colorWorkspace = nullptr;
    float* inputStatsDevice = nullptr;
    float* outputStatsDevice = nullptr;
    bool useColorTransfer = !cfg.noColorXfer;
    if (useColorTransfer) {
        cudaMalloc(&colorWorkspace, 1024 * 6 * sizeof(float));
        cudaMalloc(&inputStatsDevice, 6 * sizeof(float));
        cudaMalloc(&outputStatsDevice, 6 * sizeof(float));
        std::cerr << "Color transfer: enabled (preserves input color/brightness)" << std::endl;
    }

    // -----------------------------------------------------------------------
    // 7. Pipeline loop
    // -----------------------------------------------------------------------
    auto startTime = std::chrono::high_resolution_clock::now();
    int nFrameDecoded = 0;
    int nFrameEncoded = 0;
    uint8_t* pVideo = nullptr;
    uint8_t** ppFrame = nullptr;
    int nFrameReturned = 0;
    int nVideoBytes = 0;
    int nAudioPackets = 0;

    cudaStream_t inferStream = streams[0];

    // ---- Profiling: CUDA events for per-stage GPU timing ----
    cudaEvent_t evStart, evAfterCsc1, evAfterInfer, evAfterCsc2;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evAfterCsc1);
    cudaEventCreate(&evAfterInfer);
    cudaEventCreate(&evAfterCsc2);

    // Previous frame's timing events (for deferred readout)
    cudaEvent_t prevEvStart, prevEvAfterCsc1, prevEvAfterInfer, prevEvAfterCsc2;
    cudaEventCreate(&prevEvStart);
    cudaEventCreate(&prevEvAfterCsc1);
    cudaEventCreate(&prevEvAfterInfer);
    cudaEventCreate(&prevEvAfterCsc2);

    // Accumulators (milliseconds)
    double totalDemuxMs   = 0.0;
    double totalDecodeMs  = 0.0;
    double totalCsc1Ms    = 0.0;  // NV12->RGB
    double totalInferMs   = 0.0;
    double totalCsc2Ms    = 0.0;  // RGB->NV12
    double totalSyncMs    = 0.0;  // event sync wait
    double totalEncodeMs  = 0.0;
    double totalWriteMs   = 0.0;

    auto reportProgress = [&]() {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - startTime).count();
        double fpsNow = (elapsed > 0) ? nFrameEncoded / elapsed : 0.0;
        fprintf(stderr, "\rFrame %d | %.1f fps | %.1fs elapsed",
                nFrameEncoded, fpsNow, elapsed);
    };

    // Deferred encoding state: encode previous frame while GPU processes current
    bool hasPendingEncode = false;
    cudaEvent_t pendingDoneEvent;
    cudaEventCreate(&pendingDoneEvent);

    // Helper: encode the pending frame (called when previous GPU work is done)
    auto encodePendingFrame = [&]() {
        if (!hasPendingEncode) return;

        // Wait for previous frame's CSC2 to finish
        auto syncStart = std::chrono::high_resolution_clock::now();
        cudaEventSynchronize(pendingDoneEvent);
        auto syncEnd = std::chrono::high_resolution_clock::now();
        totalSyncMs += std::chrono::duration<double, std::milli>(syncEnd - syncStart).count();

        // Read previous frame's GPU stage timings
        float csc1Ms = 0, inferMs = 0, csc2Ms = 0;
        cudaEventElapsedTime(&csc1Ms, prevEvStart, prevEvAfterCsc1);
        cudaEventElapsedTime(&inferMs, prevEvAfterCsc1, prevEvAfterInfer);
        cudaEventElapsedTime(&csc2Ms, prevEvAfterInfer, prevEvAfterCsc2);
        totalCsc1Ms  += csc1Ms;
        totalInferMs += inferMs;
        totalCsc2Ms  += csc2Ms;

        // Encode previous frame (NVENC ASIC runs in parallel with CUDA cores)
        auto encStart = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<uint8_t>> vPacket;
        pEnc->EncodeFrame(vPacket);
        auto encEnd = std::chrono::high_resolution_clock::now();
        totalEncodeMs += std::chrono::duration<double, std::milli>(encEnd - encStart).count();

        // Write encoded packets
        auto writeStart = std::chrono::high_resolution_clock::now();
        for (auto& packet : vPacket) {
            writer.pushVideoPacket(packet.data(), (int)packet.size(), nFrameEncoded);
            nFrameEncoded++;
        }
        auto writeEnd = std::chrono::high_resolution_clock::now();
        totalWriteMs += std::chrono::duration<double, std::milli>(writeEnd - writeStart).count();

        hasPendingEncode = false;
    };

    // Lambda: process decoded frames through inference and encoding
    auto processDecodedFrames = [&](uint8_t** ppFrame, int nFrameReturned) {
        for (int i = 0; i < nFrameReturned; i++) {
            nFrameDecoded++;
            int bufIdx = nFrameDecoded % NUM_BUFFERS;
            FrameBuffer& buf = buffers[bufIdx];

            // Encode the PREVIOUS frame while we start GPU work on the current one.
            // This overlaps NVENC (hardware encoder ASIC) with CUDA compute.
            encodePendingFrame();

            // -- Color convert: NV12/P010 -> planar RGB FP16 --
            int decPitch = getDecoderPitch();
            cudaEventRecord(evStart, inferStream);
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
            cudaEventRecord(evAfterCsc1, inferStream);

            // -- Color transfer: capture input stats before inference (async) --
            if (useColorTransfer) {
                launchComputeChannelStats(
                    buf.rgbIn, srcWidth, srcHeight, padW, padH,
                    inputStatsDevice, colorWorkspace, inferStream);
            }

            // -- TRT inference (with CUDA graph capture for launch overhead reduction) --
            if (!trt.inferWithGraph(buf.rgbIn, buf.rgbOut, inferStream, bufIdx)) {
                std::cerr << "\nInference failed at frame " << nFrameDecoded << std::endl;
                return;
            }

            // -- Color transfer: match output color/brightness to input (async) --
            if (useColorTransfer) {
                launchComputeChannelStats(
                    buf.rgbOut, srcWidth, srcHeight, padW, padH,
                    outputStatsDevice, colorWorkspace, inferStream);
                launchApplyColorTransfer(
                    buf.rgbOut, srcWidth, srcHeight, padW, padH,
                    inputStatsDevice, outputStatsDevice, inferStream);
            }

            cudaEventRecord(evAfterInfer, inferStream);

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
            cudaEventRecord(evAfterCsc2, inferStream);

            // Record completion event and defer encoding to next iteration
            cudaEventRecord(pendingDoneEvent, inferStream);

            // Swap timing events so we can read them next iteration
            std::swap(evStart, prevEvStart);
            std::swap(evAfterCsc1, prevEvAfterCsc1);
            std::swap(evAfterInfer, prevEvAfterInfer);
            std::swap(evAfterCsc2, prevEvAfterCsc2);

            hasPendingEncode = true;

            // Progress
            if (nFrameDecoded % 30 == 0) {
                reportProgress();
            }
        }
    };

    // ---- Main demux loop: route video to decoder, audio/subs to muxer ----
    int streamIdx = 0;
    while (true) {
        auto demuxStart = std::chrono::high_resolution_clock::now();
        bool gotPacket = demuxer.DemuxAny(&pVideo, &nVideoBytes, &streamIdx);
        auto demuxEnd = std::chrono::high_resolution_clock::now();
        totalDemuxMs += std::chrono::duration<double, std::milli>(demuxEnd - demuxStart).count();

        if (!gotPacket) break;

        if (streamIdx == demuxer.GetVideoStreamIndex()) {
            // Video packet -> decoder (NVDEC or software)
            auto decodeStart = std::chrono::high_resolution_clock::now();
            if (nvDec) {
                nvDec->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
            } else {
                swDec->Decode(pVideo, nVideoBytes, &ppFrame, &nFrameReturned);
            }
            auto decodeEnd = std::chrono::high_resolution_clock::now();
            totalDecodeMs += std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

            processDecodedFrames(ppFrame, nFrameReturned);
        } else if (passAudio) {
            // Audio/subtitle packet -> passthrough to muxer via async writer
            AVPacket* pkt = demuxer.GetCurrentPacket();
            AVStream* s = demuxer.GetStream(streamIdx);
            if (s && (s->codecpar->codec_type == AVMEDIA_TYPE_AUDIO ||
                      s->codecpar->codec_type == AVMEDIA_TYPE_SUBTITLE))
            {
                writer.pushPassthroughPacket(pkt);
                nAudioPackets++;
            }
        }
    }

    // Flush decoder: send empty packet to get buffered frames
    {
        auto decodeStart = std::chrono::high_resolution_clock::now();
        if (nvDec) {
            nvDec->Decode(nullptr, 0, &ppFrame, &nFrameReturned);
        } else {
            swDec->Decode(nullptr, 0, &ppFrame, &nFrameReturned);
        }
        auto decodeEnd = std::chrono::high_resolution_clock::now();
        totalDecodeMs += std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();
    }
    processDecodedFrames(ppFrame, nFrameReturned);

    // Encode the last pending frame
    encodePendingFrame();

    // -----------------------------------------------------------------------
    // 8. Flush encoder
    // -----------------------------------------------------------------------
    {
        auto encStart = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<uint8_t>> vPacket;
        pEnc->EndEncode(vPacket);
        auto encEnd = std::chrono::high_resolution_clock::now();
        totalEncodeMs += std::chrono::duration<double, std::milli>(encEnd - encStart).count();
        for (auto& packet : vPacket) {
            writer.pushVideoPacket(packet.data(), (int)packet.size(), nFrameEncoded);
            nFrameEncoded++;
        }
    }

    // -----------------------------------------------------------------------
    // 9. Flush writer and close muxer
    // -----------------------------------------------------------------------
    writer.stop();   // waits for all queued packets to be written
    muxer.close();   // writes MKV trailer

    // -----------------------------------------------------------------------
    // 10. Report results and clean up
    // -----------------------------------------------------------------------
    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();
    double avgFps = nFrameEncoded / totalTime;

    fprintf(stderr, "\n\nDone: %d frames in %.2f seconds (%.1f fps avg)\n",
            nFrameEncoded, totalTime, avgFps);
    fprintf(stderr, "Output: %s\n", cfg.outputPath.c_str());
    if (passAudio) {
        fprintf(stderr, "Passthrough: %d audio/subtitle packets\n", nAudioPackets);
    }

    // ---- Per-stage profiling breakdown ----
    if (nFrameDecoded > 0) {
        double totalAccountedMs = totalDemuxMs + totalDecodeMs + totalCsc1Ms
                                + totalInferMs + totalCsc2Ms + totalSyncMs
                                + totalEncodeMs + totalWriteMs;
        double wallMs = totalTime * 1000.0;
        double overheadMs = wallMs - totalAccountedMs;

        fprintf(stderr, "\n--- Per-stage timing breakdown (%d frames) ---\n", nFrameDecoded);
        fprintf(stderr, "  Demux (CPU):          %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                totalDemuxMs,  totalDemuxMs / nFrameDecoded,  100.0*totalDemuxMs/wallMs);
        fprintf(stderr, "  Decode (%s):  %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                nvDec ? "NVDEC+CPU" : "SW+upload",
                totalDecodeMs, totalDecodeMs / nFrameDecoded, 100.0*totalDecodeMs/wallMs);
        fprintf(stderr, "  CSC NV12->RGB (GPU):  %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                totalCsc1Ms,   totalCsc1Ms / nFrameDecoded,   100.0*totalCsc1Ms/wallMs);
        fprintf(stderr, "  TRT Infer (GPU):      %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                totalInferMs,  totalInferMs / nFrameDecoded,  100.0*totalInferMs/wallMs);
        fprintf(stderr, "  CSC RGB->NV12 (GPU):  %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                totalCsc2Ms,   totalCsc2Ms / nFrameDecoded,   100.0*totalCsc2Ms/wallMs);
        fprintf(stderr, "  Stream sync (CPU):    %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                totalSyncMs,   totalSyncMs / nFrameDecoded,   100.0*totalSyncMs/wallMs);
        fprintf(stderr, "  Encode (NVENC):       %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                totalEncodeMs, totalEncodeMs / nFrameDecoded, 100.0*totalEncodeMs/wallMs);
        fprintf(stderr, "  Queue push (async):   %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                totalWriteMs,  totalWriteMs / nFrameDecoded,  100.0*totalWriteMs/wallMs);
        fprintf(stderr, "  Overhead/unaccounted: %8.1f ms total  %6.3f ms/frame  %5.1f%%\n",
                overheadMs,    overheadMs / nFrameDecoded,    100.0*overheadMs/wallMs);
        fprintf(stderr, "  ---\n");
        fprintf(stderr, "  Wall time:            %8.1f ms total  %6.3f ms/frame\n",
                wallMs, wallMs / nFrameDecoded);
        fprintf(stderr, "  GPU work (CSC+TRT):   %8.1f ms total  %6.3f ms/frame (theoretical max: %.1f fps)\n",
                totalCsc1Ms + totalInferMs + totalCsc2Ms,
                (totalCsc1Ms + totalInferMs + totalCsc2Ms) / nFrameDecoded,
                1000.0 / ((totalCsc1Ms + totalInferMs + totalCsc2Ms) / nFrameDecoded));
        fprintf(stderr, "  Sequential overhead:  %6.3f ms/frame (%.1f fps lost vs GPU-only)\n",
                (wallMs / nFrameDecoded) - ((totalCsc1Ms + totalInferMs + totalCsc2Ms) / nFrameDecoded),
                avgFps > 0 ? (1000.0 / ((totalCsc1Ms + totalInferMs + totalCsc2Ms) / nFrameDecoded) - avgFps) : 0.0);
    }

    // Cleanup profiling events
    cudaEventDestroy(evStart);
    cudaEventDestroy(evAfterCsc1);
    cudaEventDestroy(evAfterInfer);
    cudaEventDestroy(evAfterCsc2);
    cudaEventDestroy(prevEvStart);
    cudaEventDestroy(prevEvAfterCsc1);
    cudaEventDestroy(prevEvAfterInfer);
    cudaEventDestroy(prevEvAfterCsc2);
    cudaEventDestroy(pendingDoneEvent);

    // Free color transfer buffers
    if (colorWorkspace) cudaFree(colorWorkspace);
    if (inputStatsDevice) cudaFree(inputStatsDevice);
    if (outputStatsDevice) cudaFree(outputStatsDevice);

    // Free pipeline buffers
    for (int i = 0; i < NUM_BUFFERS; i++) {
        buffers[i].free();
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(events[i]);
    }

    return 0;

    } catch (const std::exception& ex) {
        std::cerr << "\nFatal error: " << ex.what() << std::endl;
        return 1;
    }
}
