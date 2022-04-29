#include <algorithm>
#include <numeric>
#include <fstream>
#include <memory>
#include <iomanip>
#include <chrono>

#include "Utils.h"
#include "argparse.h"
#include "Log.h"

#if defined(ENABLE_PYTORCH_FRONTEND)
#include "PytorchFrontend.h"
#elif defined(ENABLE_TFLITE_FRONTEND)
#include "TFLiteFrontend.h"
#elif defined(ENABLE_ARMNN_FRONTEND)
#include "ArmNNFrontend.h"
#elif defined(ENABLE_RTNEURAL_FRONTEND)
#include "RTNeuralFrontend.h"
#endif

int main(int argc, char *argv[])
{
    argparse::ArgumentParser program("Benchmark for diverse frontends.");

    program.add_argument("-m", "--model").required().help("Input.");
    program.add_argument("-p", "--sequence_lenght")
        .required()
        .scan<'i', int>();

    program.add_argument("--tflite_with_armnn_delegate")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("-f", "--frontend")
        .required()
        .default_value(0)
        .scan<'i', int>();

    program.add_argument("--write_debug")
        .default_value(false)
        .implicit_value(true);

    program.add_argument("--average_iterations")
        .default_value(100)
        .scan<'i', int>();

    try
    {
        program.parse_args(argc, argv);
    }
    catch (const std::runtime_error &err)
    {
        NN_LOG(ERROR) << err.what();
        NN_LOG(ERROR) << program;
        std::exit(1);
    }

    auto modelName = program.get<std::string>("-m");
    int sequenceLenght = program.get<int>("sequence_lenght");
    int frontend = program.get<int>("frontend");
    unsigned int iterations = program.get<int>("average_iterations");
    bool useTFLiteArmNNDelegate = program["--tflite_with_armnn_delegate"] == true;
    bool writeDebug = program["--write_debug"] == true;

    NN_LOG(INFO) << "Model name " << modelName;
    NN_LOG(INFO) << "Sequence lenght " << sequenceLenght;

#if defined(ENABLE_PYTORCH_FRONTEND)
    NN_LOG(INFO) << "Creating Pytorch pipeline";
    PytorchFrontend nn;
#elif defined(ENABLE_TFLITE_FRONTEND)
    NN_LOG(INFO) << "Creating TFLite pipeline";
    if (useTFLiteArmNNDelegate)
    {
        NN_LOG(INFO) << "With ArmNN delegate";
    }
    auto nn = std::make_unique<TFLiteFrontend>(useTFLiteArmNNDelegate);
#elif defined(ENABLE_ARMNN_FRONTEND)
    NN_LOG(INFO) << "Creating ArmNN pipeline";
    auto nn = std::make_unique<ArmNNFrontend>();
#elif defined(ENABLE_RTNEURAL_FRONTEND)
    NN_LOG(INFO) << "Creating RTNeural pipeline";
    auto nn = std::make_unique<RTNeuralFrontend>();
#endif

    // if (!nn)
    // {
        // NN_LOG(ERROR) << "Could not create the frontend";
        // std::exit(1);
    // }

    nn.load(modelName);

    std::vector<float> in = Utils::linspace(-1.0F, 1.0F, sequenceLenght);
    std::vector<float> out(sequenceLenght, 0);
    std::vector<double> durations;

    for (unsigned int i = 0; i < iterations; i++)
    {
        const auto start_time = std::chrono::high_resolution_clock::now();
        nn.process(in, out);
        const auto duration = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - start_time);
        durations.push_back(duration.count());
    }

    for (unsigned int i = 0; i < 10; i++)
    {
        NN_LOG(INFO) << std::fixed << std::setprecision(3) << "In: " << in[i] << "    Out: " << out[i];
    }

    float average = std::accumulate(std::begin(durations), std::end(durations), 0.0) / durations.size();

    NN_LOG(INFO) << "Inference done, in average after " << iterations << " iterations :" << std::fixed << average << "ms";

    if (writeDebug)
    {
        std::ofstream debug_out("out.txt");
        std::copy(out.begin(), out.end(), std::ostream_iterator<float>(debug_out, "\n"));
    }
}