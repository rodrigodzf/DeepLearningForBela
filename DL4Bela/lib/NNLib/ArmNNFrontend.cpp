#include <armnnOnnxParser/IOnnxParser.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <filesystem>
#include "ArmNNFrontend.h"
#include "Utils.h"

using namespace armnn;
namespace fs = std::filesystem;

ArmNNFrontend::ArmNNFrontend()
    : mRuntime(IRuntime::Create(IRuntime::CreationOptions()))
{
}

ArmNNFrontend::~ArmNNFrontend()
{
}

bool ArmNNFrontend::load(const std::string &filename)
{
    // Get the filename extension
    auto extension = fs::path(filename).extension();
    bool isTFLite = false;
    if (extension == ".tflite")
    {
        isTFLite = true;
    }

    armnn::INetworkPtr network = INetwork::Create();

    // Construct ArmNN network
    if (isTFLite)
    {
        NN_LOG(INFO) << "Loading TFLite Model " << filename;
        armnnTfLiteParser::ITfLiteParser::TfLiteParserOptions tfLiteParserOptions;
        tfLiteParserOptions.m_InferAndValidate = true;

        auto parser = armnnTfLiteParser::ITfLiteParser::Create(tfLiteParserOptions);
        network = parser->CreateNetworkFromBinaryFile(filename.c_str());
        NN_LOG(INFO) << "Loaded: " << filename;

        auto inputTensorNames = parser->GetSubgraphInputTensorNames(0);
        auto outputTensorNames = parser->GetSubgraphOutputTensorNames(0);
        for (auto const &name : inputTensorNames)
        {
            NN_LOG(INFO) << "Input names " << name;
        }

        for (auto const &name : outputTensorNames)
        {
            NN_LOG(INFO) << "output names " << name;
        }

        // Find the binding points for the input and output nodes
        mInputBindingInfo = parser->GetNetworkInputBindingInfo(0, inputTensorNames[0]);
        mOutputBindingInfo = parser->GetNetworkOutputBindingInfo(0, outputTensorNames[0]);

        NN_LOG(INFO) << "Input shape " << mInputBindingInfo.second.GetShape();
        NN_LOG(INFO) << "Output shape " << mOutputBindingInfo.second.GetShape();

        NN_LOG(INFO) << "Binding points set";
    }
    else
    {
        NN_LOG(INFO) << "Loading ONNX Model " << filename;
        auto parser = armnnOnnxParser::IOnnxParser::Create();
        network = parser->CreateNetworkFromBinaryFile(filename.c_str());
        NN_LOG(INFO) << "Loaded: " << filename;

        // Find the binding points for the input and output nodes
        mInputBindingInfo = parser->GetNetworkInputBindingInfo("input");
        mOutputBindingInfo = parser->GetNetworkOutputBindingInfo("output");

        NN_LOG(INFO) << "Binding points set";
    }

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
    IRuntime::CreationOptions options;
    auto optimizerOptions = armnn::OptimizerOptions();
    optimizerOptions.m_shapeInferenceMethod = armnn::ShapeInferenceMethod::InferAndValidate;

    // runtime(IRuntime::Create(options));
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    try
    {
        optNet = armnn::Optimize(*network,
                                 {armnn::Compute::CpuAcc},
                                 mRuntime->GetDeviceSpec(),
                                 optimizerOptions);
    }
    catch (std::exception &ex)
    {
        std::stringstream exMessage;
        exMessage << "Exception (" << ex.what() << ") caught from optimize.";
        throw armnn::Exception(exMessage.str());
    }

    if (!optNet)
    {
        // Optimize failed
        throw armnn::Exception("Unable to optimize the network!");
        std::exit(1);
    }

    NN_LOG(INFO) << "Optimized";

    // Load the optimized network onto the runtime device
    std::string errorMessage;
    if (Status::Success != mRuntime->LoadNetwork(mNetworkIdentifier, std::move(optNet), errorMessage))
    {
        NN_LOG(ERROR) << errorMessage;
        std::exit(1);
        return false;
    }

    NN_LOG(INFO) << "Network loaded";

    // pre-allocate memory for output (the size of it never changes)
    // mOutputBuffer.assign(mOutputBindingInfo.second.GetNumElements(), 0);
    // mOutputTensors = MakeOutputTensors(mOutputBindingInfo, mOutputBuffer.data());

    return true;
}