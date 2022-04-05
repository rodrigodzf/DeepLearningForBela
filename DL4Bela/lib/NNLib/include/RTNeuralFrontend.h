/**
  \ingroup NNLib
  \file    RTNeuralFrontend
  \brief   This file contains the implementation for class RTNeuralFrontend.
  \author  rodrigodzf@gmail.com
  \date    2022-03-28
*/

#pragma once

#include <string>
#include <vector>
#include <array>
#include <memory>

#include "BaseNN.h"
#ifdef RTNEURAL_STATIC
#include <RTNeural/RTNeural.h>
#endif
// Fwd
namespace RTNeural
{
#ifdef RTNEURAL_DYNAMIC
    using MLP = Model<float>;
#elif RTNEURAL_STATIC
    using MLP = ModelT<float, 1, 1,
                       DenseT<float, 1, 128>,
                       ReLuActivationT<float, 128>,
                       DenseT<float, 128, 128>,
                       ReLuActivationT<float, 128>,
                       DenseT<float, 128, 1>>;
#endif
} // namespace RTNeural

class RTNeuralFrontend : public BaseNN
{
private:
#if defined(RTNEURAL_DYNAMIC) || defined(RTNEURAL_STATIC)
    std::unique_ptr<RTNeural::MLP> m_model;
#endif

public:
    RTNeuralFrontend();
    ~RTNeuralFrontend();

    bool load(const std::string &filename) override;
    inline bool process(const std::vector<float> &inputData,
                        std::vector<float> &outResults) override final;
};
