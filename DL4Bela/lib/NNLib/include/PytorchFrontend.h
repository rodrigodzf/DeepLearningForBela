/**
  \ingroup NNLib
  \file    PytorchFrontend
  \brief   This file contains the implementation for class PytorchFrontend.
  \author  rodrigodzf@gmail.com
  \date    2022-03-28
*/

#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>

#include "BaseNN.h"
#include <torch/script.h>


class PytorchFrontend : public BaseNN
{
private:
    torch::jit::script::Module mModule;
private:
    void printDebug();

public:
    PytorchFrontend();
    ~PytorchFrontend();

    bool load(const std::string &filename) override;

    template <typename T>
    bool process(std::vector<T> &inputData,
                 std::vector<T> &outResults)
    {
        c10::InferenceMode guard;

        at::Tensor input = torch::from_blob(inputData.data(), {inputData.size(), 1}, at::kFloat);

        auto output = mModule.forward({input}).toTensor();

        std::memcpy(outResults.data(), output.data_ptr<float>(), outResults.size() * sizeof(T));

        return true;
    }
};
