#include <RTNeural/RTNeural.h>
#include "RTNeuralFrontend.h"
#include "Utils.h"
#include "Log.h"

RTNeuralFrontend::RTNeuralFrontend()
{
}

RTNeuralFrontend::~RTNeuralFrontend()
{
}

bool RTNeuralFrontend::load(const std::string &filename)
{
    std::ifstream jsonStream(filename, std::ifstream::binary);

#ifdef RTNEURAL_STATIC
    // define static model    
    m_model = std::make_unique<RTNeural::MLP>();
    m_model->parseJson(jsonStream, true);
#elif RTNEURAL_DYNAMIC
    m_model = RTNeural::json_parser::parseJson<float>(jsonStream, true);
#endif
    m_model->reset();
    return true;
}

bool RTNeuralFrontend::process(const std::vector<float> &inputData,
                         std::vector<float> &outResults)
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