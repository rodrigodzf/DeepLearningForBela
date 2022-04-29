#include "PytorchFrontend.h"

#include "Log.h"

PytorchFrontend::PytorchFrontend()
{}

PytorchFrontend::~PytorchFrontend()
{
}

void PytorchFrontend::printDebug()
{

}

bool PytorchFrontend::load(const std::string &filename)
{
    c10::InferenceMode guard;
    torch::jit::setGraphExecutorOptimize(false);
    torch::jit::getProfilingMode() = false;
    try
    {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        mModule = torch::jit::load(filename);
        mModule.eval();

    }
    catch (const std::exception &e)
    {
        NN_LOG(ERROR) << "error loading the model";
        std::exit(1);
    }

    // debug
    printDebug();

    return true;
}
