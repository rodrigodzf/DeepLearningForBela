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
#include <RTNeural/RTNeural.h>
// Fwd
// namespace RTNeural
// {
#ifdef RTNEURAL_DYNAMIC
    using MLP = RTNeural::Model<float>;
// #elif RTNEURAL_STATIC
//     using MLP = ModelT<float, 1, 1,
//                        DenseT<float, 1, 128>,
//                        ReLuActivationT<float, 128>,
//                        DenseT<float, 128, 128>,
//                        ReLuActivationT<float, 128>,
//                        DenseT<float, 128, 1>>;
#endif
// } // namespace RTNeural

class RTNeuralFrontend : public BaseNN
{
private:
#if defined(RTNEURAL_DYNAMIC) || defined(RTNEURAL_STATIC)
    std::unique_ptr<MLP> m_model;
#endif

public:
    RTNeuralFrontend();
    ~RTNeuralFrontend();

    bool load(const std::string &filename) override;
    
    template <typename T>
    bool process(const std::vector<T> &inputData,
                 std::vector<T> &outResults)
    {

        for (unsigned int i = 0; i < inputData.size(); i++)
        {
            outResults[i] = m_model->forward(&inputData[i]);
        }

#if 0
    float first = m_model->forward(inputData.data());
    auto out = m_model->getOutputs();
    std::copy(out, out + outResults.size(), outResults.begin());
#endif

        return true;
    }
};
