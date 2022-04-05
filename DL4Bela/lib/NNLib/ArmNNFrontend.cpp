#include <armnnOnnxParser/IOnnxParser.hpp>
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#include <filesystem>
#include "ArmNNFrontend.h"
#include "Utils.h"
#include "Log.h"

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

    // auto network = mParser->CreateNetworkFromBinaryFile(filename, inputShapes, requestedOutputs);
    armnn::INetworkPtr network = INetwork::Create();

    // Construct ArmNN network
    if (isTFLite)
    {
        LOG(INFO) << "Loading TFLite Model " << filename;
        auto parser = armnnTfLiteParser::ITfLiteParser::Create();
        network = parser->CreateNetworkFromBinaryFile(filename.c_str());
        LOG(INFO) << "Loaded: " << filename;

        auto inputTensorNames = parser->GetSubgraphInputTensorNames(0);
        auto outputTensorNames = parser->GetSubgraphOutputTensorNames(0);
        for (auto const &name : inputTensorNames)
        {
            LOG(INFO) << "Input names " << name;
        }

        for (auto const &name : outputTensorNames)
        {
            LOG(INFO) << "output names " << name;
        }

        // Find the binding points for the input and output nodes
        auto inputBindingInfo = parser->GetNetworkInputBindingInfo(0, inputTensorNames[0]);
        auto outputBindingInfo = parser->GetNetworkOutputBindingInfo(0, outputTensorNames[0]);
        LOG(INFO) << "Binding points set";
    }
    else
    {
        LOG(INFO) << "Loading ONNX Model " << filename;
        auto parser = armnnOnnxParser::IOnnxParser::Create();
        network = parser->CreateNetworkFromBinaryFile(filename.c_str());
        LOG(INFO) << "Loaded: " << filename;

        // Find the binding points for the input and output nodes
        mInputBindingInfo = parser->GetNetworkInputBindingInfo("input");
        mOutputBindingInfo = parser->GetNetworkOutputBindingInfo("output");

        LOG(INFO) << "Binding points set";
    }

    // Optimize the network for a specific runtime compute device, e.g. CpuAcc, GpuAcc
    IRuntime::CreationOptions options;
    // runtime(IRuntime::Create(options));
    armnn::IOptimizedNetworkPtr optNet(nullptr, nullptr);
    try
    {
        optNet = armnn::Optimize(*network, {armnn::Compute::CpuAcc}, mRuntime->GetDeviceSpec());
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

    LOG(INFO) << "Optimized";

    // Load the optimized network onto the runtime device
    std::string errorMessage;
    if (Status::Success != mRuntime->LoadNetwork(mNetworkIdentifier, std::move(optNet), errorMessage))
    {
        LOG(ERROR) << errorMessage;
        std::exit(1);
        return false;
    }

    LOG(INFO) << "Network loaded";

    // pre-allocate memory for output (the size of it never changes)
    mOutputBuffer.assign(mOutputBindingInfo.second.GetNumElements(), 0);
    mOutputTensors = MakeOutputTensors(mOutputBindingInfo, mOutputBuffer.data());

    return true;
}

bool ArmNNFrontend::process(const std::vector<float> &inputData,
                          std::vector<float> &outResults)
{

    mInputTensors = MakeInputTensors(mInputBindingInfo, inputData.data());

    // enqueue workload
    Status ret = mRuntime->EnqueueWorkload(mNetworkIdentifier,
                                           mInputTensors,
                                           mOutputTensors);

    if (ret == Status::Failure)
    {
        LOG(ERROR) << "Failed to perform inference.";
        return false;
    }

    outResults = mOutputBuffer;
    return true;
}