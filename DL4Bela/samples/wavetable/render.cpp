// Queen Mary University of London
// ECS7012 - Music and Audio Programming
// Spring 2022
//
// Assignment 1: Synth Filter
// This project contains template code for implementing a resonant filter with
// parameters adjustable in the Bela GUI, and waveform visible in the scope

#include <Bela.h>
#include <libraries/Gui/Gui.h>
#include <libraries/GuiController/GuiController.h>
#include <libraries/Scope/Scope.h>
#include <libraries/math_neon/math_neon.h>
#include <cmath>
#include <map>
#include <iostream>
#include <algorithm>
#include "Wavetable.h"

#include "BaseNN.h"
#include "ArmNNFrontend.h"
#include "TFLiteFrontend.h"
#include "AppOptions.h"
#include "Log.h"

// Browser-based GUI to adjust parameters
Gui gGui;
GuiController gGuiController;
Wavetable gSawtoothOscillator;

// Browser-based oscilloscope to visualise signal
// Scope gScope;
// Oscillator objects

// ArmNNFrontend
std::unique_ptr<BaseNN> nn;

// buffers
std::vector<float> gIn;
std::vector<float> gOut;

bool setup(BelaContext *context, void *userData)
{
    AppOptions *opts = reinterpret_cast<AppOptions *>(userData);

    std::cout << "Frame size: " << context->audioFrames << std::endl;
    std::cout << "Sample rate: " << context->audioSampleRate << std::endl;

    std::vector<float> wavetable;
    const unsigned int wavetableSize = 1024;
    wavetable.resize(wavetableSize);

    // Generate a sawtooth wavetable (a ramp from -1 to 1)
    for (unsigned int n = 0; n < wavetableSize; n++)
    {
        wavetable[n] = -1.0 + 2.0 * (float)n / (float)(wavetableSize - 1);
    }
    // Initialise the sawtooth wavetable, passing the sample rate and the buffer
    gSawtoothOscillator.setup(context->audioSampleRate, wavetable);

    // Set up ArmNNFrontend
    switch (opts->frontend)
    {
    case 0:
#if defined(ENABLE_TFLITE_FRONTEND)
        LOG(INFO) << "Creating TFLite pipeline";
        nn = std::make_unique<TFLiteFrontend>(false);
#endif
        break;
    case 1:
#if defined(ENABLE_ARMNN_FRONTEND)
        LOG(INFO) << "Creating ArmNN pipeline";
        nn = std::make_unique<ArmNNFrontend>();
#endif
        break;
    case 2:
#if defined(ENABLE_RTNEURAL_FRONTEND)
        LOG(INFO) << "Creating RTNeural pipeline";
        nn = std::make_unique<RTNeuralFrontend>();
#endif
        break;
    default:
        break;
    }

    if (!nn)
    {
        LOG(ERROR) << "Could not create the frontend";
        std::exit(1);
    }

    bool ret = nn->load(opts->modelPath);

    // reserve buffers
    gIn.resize(context->audioFrames);
    gIn.reserve(context->audioFrames);

    gOut.resize(context->audioFrames);
    gOut.reserve(context->audioFrames);

    // Set up the GUI
    gGui.setup(context->projectName);
    gGuiController.setup(&gGui, "Oscillator and Filter Controls");

    // Arguments: name, default value, minimum, maximum, increment
    // Create sliders for oscillator and filter settings
    gGuiController.addSlider("Oscillator Frequency", 100, 1, 1000, 1);
    gGuiController.addSlider("Oscillator Amplitude", 0.5, 0, 1.0, 0);
    gGuiController.addSlider("Out Amplitude", 0.01, 0, 1.0, 0);

    // Set up the scope
    // gScope.setup(2, context->audioSampleRate);
    return true;
}

void render(BelaContext *context, void *userData)
{
    // Read the slider values
    float oscFrequency = gGuiController.getSliderValue(0);
    float oscAmplitude = gGuiController.getSliderValue(1);
    float outAmplitude = gGuiController.getSliderValue(2);

    gSawtoothOscillator.setFrequency(oscFrequency);

    for (unsigned int n = 0; n < context->audioFrames; n++)
    {
        gIn[n] = oscAmplitude * gSawtoothOscillator.process();
    }

    // process ArmNNFrontend
    nn->process(gIn, gOut);

    // Write the output to every audio channel
    for (unsigned int n = 0; n < context->audioFrames; n++)
    {
        audioWrite(context, n, 0, outAmplitude * gIn[n]);
        audioWrite(context, n, 1, outAmplitude * gOut[n]);
    }
}

void cleanup(BelaContext *context, void *userData)
{
}
