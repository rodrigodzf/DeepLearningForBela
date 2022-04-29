/**
  \ingroup NNLib
  \file    ArmNNFrontend
  \brief   This file contains the implementation for class ArmNNFrontend.
  \author  rodrigodzf@gmail.com
  \date    2022-03-28
*/

#pragma once

#include <string>
#include <armnn/INetwork.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/Descriptors.hpp>

#include "BaseNN.h"
#include "Log.h"

class ArmNNFrontend : public BaseNN
{
private:
    armnn::NetworkId mNetworkIdentifier{};
    armnn::IRuntimePtr mRuntime;
    armnn::InputTensors mInputTensors;
    armnn::OutputTensors mOutputTensors;
    armnn::BindingPointInfo mInputBindingInfo;
    armnn::BindingPointInfo mOutputBindingInfo;

    // ! TODO template
    std::vector<int8_t> mOutputBuffer;

private:
    // Helper function to make input tensors
    inline armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId,
                                                                armnn::TensorInfo> &input,
                                                const void *inputTensorData)
    {
        return {{input.first, armnn::ConstTensor(input.second, inputTensorData)}};
    }

    // Helper function to make output tensors
    inline armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId,
                                                                  armnn::TensorInfo> &output,
                                                  void *outputTensorData)
    {
        return {{output.first, armnn::Tensor(output.second, outputTensorData)}};
    }

public:
    float getQuantizationScale()
    {
        return mInputBindingInfo.second.GetQuantizationScale();
    }

    int getQuantizationOffset()
    {
        return mInputBindingInfo.second.GetQuantizationOffset();
    }

public:
    ArmNNFrontend();
    ~ArmNNFrontend();

    bool load(const std::string &filename) override;
    // inline bool process(const std::vector<float> &inputData,
                        // std::vector<float> &outResults) override final;
    
    template <typename T>
    inline bool process(const std::vector<T> &inputData,
                        std::vector<T> &outResults)
    {

        mInputTensors = MakeInputTensors(mInputBindingInfo, inputData.data());
        mOutputTensors = MakeOutputTensors(mOutputBindingInfo, outResults.data());

        // enqueue workload
        auto ret = mRuntime->EnqueueWorkload(mNetworkIdentifier,
                                            mInputTensors,
                                            mOutputTensors);

        if (ret == armnn::Status::Failure)
        {
            NN_LOG(ERROR) << "Failed to perform inference.";
            return false;
        }

        return true;
    }
};
