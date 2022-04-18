#include "TFLiteFrontend.h"

#include <armnn_delegate.hpp>
#include <tensorflow/lite/kernels/builtin_op_kernels.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include "Log.h"

TFLiteFrontend::TFLiteFrontend(bool useArmnnDelegate)
{
    m_useArmnnDelegate = useArmnnDelegate;
}

TFLiteFrontend::~TFLiteFrontend()
{
}

void TFLiteFrontend::printDebug()
{
    LOG(INFO) << "tensors size: " << m_interpreter->tensors_size();
    LOG(INFO) << "nodes size: " << m_interpreter->nodes_size();
    LOG(INFO) << "inputs: " << m_interpreter->inputs().size();
    LOG(INFO) << "input(0) name: " << m_interpreter->GetInputName(0);
    LOG(INFO) << "number of inputs: " << m_interpreter->inputs().size();
    LOG(INFO) << "number of outputs: " << m_interpreter->outputs().size();
    // LOG(INFO) << "input dims: " << dims->size;
    // LOG(INFO) << "output dims: " << out_dims->size;

    int input = m_interpreter->inputs()[0];
    int output = m_interpreter->outputs()[0];

    LOG(INFO) << "input total elements: " << m_interpreter->tensor(input)->bytes / sizeof(m_interpreter->tensor(input)->type);
    LOG(INFO) << "output total elements: " << m_interpreter->tensor(output)->bytes / sizeof(m_interpreter->tensor(output)->type);

    LOG(INFO) << "=== Pre-invoke Interpreter State ===";
    tflite::PrintInterpreterState(m_interpreter.get());
}

bool TFLiteFrontend::load(const std::string &filename)
{
    using namespace armnn;
    using namespace tflite;

    m_model = tflite::FlatBufferModel::BuildFromFile(filename.c_str());

    // Build the interpreter
    m_interpreter = std::make_unique<Interpreter>();
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder builder(*m_model, resolver);
    builder(&m_interpreter);

    if (m_useArmnnDelegate)
    {
        // Create the Armnn Delegate
        armnn::OptimizerOptions optimizerOptions;
        // optimizerOptions.m_ReduceFp32ToFp16 = true;
        optimizerOptions.m_ProfilingEnabled = false;

        armnnDelegate::DelegateOptions delegateOptions({armnn::Compute::CpuAcc}, optimizerOptions);

        std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
            theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                             armnnDelegate::TfLiteArmnnDelegateDelete);
        // Register armnn_delegate to m_interpreter
        auto status = m_interpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
        if (status == kTfLiteError)
        {
            LOG(ERROR) << "Could not register ArmNN TfLite Delegate to m_interpreter!";
            return EXIT_FAILURE;
        }
    }

    // Allocate tensors.
    m_interpreter->AllocateTensors();

    // debug
    printDebug();

    return true;
}

