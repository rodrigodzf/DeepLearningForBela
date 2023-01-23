#include <Bela.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "AppOptions.h"


// The model's lifetime must be at least as long as the interpreter's lifetime. In order to use the interpreter in the render loop, we need to define both model and interpreter as global variables. https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter-builder
std::unique_ptr<tflite::FlatBufferModel> model;
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;

bool setup(BelaContext *context, void *userData)
{   
    printf("analog sample rate: %.1f\n", context->analogSampleRate);
    
    AppOptions *opts = reinterpret_cast<AppOptions *>(userData);

    model = tflite::FlatBufferModel::BuildFromFile(opts->modelPath.c_str());
        if(!model){
        printf("Failed to mmap model\n");
        return false;
    }

    tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

    // AllocateTensors must be called after the interpreter has been created and before running inference and accessing tensor buffers, and must be called again if and only if an input tensor is resized. https://www.tensorflow.org/lite/api_docs/cc/class/tflite/interpreter#allocatetensors
    interpreter->AllocateTensors();
  
	return true;
}

void render(BelaContext *context, void *userData)
{
    float* input = interpreter->typed_input_tensor<float>(0);

    // Dummy input for testing
    *input = 2.0;
    rt_printf("Input is: %.2f\n", *input);

    interpreter->Invoke();

    float* output = interpreter->typed_output_tensor<float>(0);
    rt_printf("Result is: %.2f\n", *output);

}

void cleanup(BelaContext *context, void *userData)
{

}