/*
 * Adapted from https://stackoverflow.com/a/56868813
 */

#include <Bela.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "AppOptions.h"


bool setup(BelaContext *context, void *userData)
{   
    AppOptions *opts = reinterpret_cast<AppOptions *>(userData);

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(opts->modelPath.c_str());
        if(!model){
        printf("Failed to mmap model\n");
        return false;
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    // Resize input tensors, if desired.
    interpreter->AllocateTensors();

    float* input = interpreter->typed_input_tensor<float>(0);
    // Dummy input for testing
    *input = 2.0;

    interpreter->Invoke();

    float* output = interpreter->typed_output_tensor<float>(0);

    printf("Result is: %f\n", *output);

	return true;
}

void render(BelaContext *context, void *userData)
{

}

void cleanup(BelaContext *context, void *userData)
{

}