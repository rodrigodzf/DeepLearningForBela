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

class ArmNNFrontend : public BaseNN
{
private:
    armnn::NetworkId mNetworkIdentifier{};
    armnn::IRuntimePtr mRuntime;
    armnn::InputTensors mInputTensors;
    armnn::OutputTensors mOutputTensors;
    armnn::BindingPointInfo mInputBindingInfo;
    armnn::BindingPointInfo mOutputBindingInfo;

    std::vector<float> mOutputBuffer;

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
    ArmNNFrontend();
    ~ArmNNFrontend();

    bool load(const std::string &filename) override;
    inline bool process(const std::vector<float> &inputData,
                        std::vector<float> &outResults) override final;
};
