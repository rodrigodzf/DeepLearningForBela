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
    inline bool process(const std::vector<float> &inputData,
                        std::vector<float> &outResults) override final;
};
