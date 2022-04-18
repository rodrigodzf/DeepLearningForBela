/**
  \ingroup NNLib
  \file    TFLiteFrontend
  \brief   This file contains the implementation for class TFLiteFrontend.
  \author  rodrigodzf@gmail.com
  \date    2022-03-28
*/

#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>

#include "BaseNN.h"
#include <tensorflow/lite/interpreter.h>

// Fwd
namespace tflite
{
    class FlatBufferModel;
    class Interpreter;
} // namespace tflite

class TFLiteFrontend : public BaseNN
{
private:
    bool m_useArmnnDelegate = false;

private:
    void printDebug();
    std::unique_ptr<tflite::FlatBufferModel> m_model;
    std::unique_ptr<tflite::Interpreter> m_interpreter;

public:
    TFLiteFrontend(bool useArmnnDelegate);
    ~TFLiteFrontend();

    bool load(const std::string &filename) override;

    template <typename T>
    bool process(const std::vector<T> &inputData,
                 std::vector<T> &outResults)
    {
        // Get pointer to the tensors
        T *input_buffer = m_interpreter->typed_input_tensor<T>(0);
        T *output_buffer = m_interpreter->typed_output_tensor<T>(0);

        // Copy input data to tensor
        std::copy(inputData.begin(), inputData.end(), input_buffer);

        m_interpreter->Invoke();

        // Copy tensor data to the output
        std::copy(output_buffer, output_buffer + outResults.size(), outResults.begin());

        return true;
    }
};
