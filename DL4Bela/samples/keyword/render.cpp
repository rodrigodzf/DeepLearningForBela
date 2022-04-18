/*
 ____  _____ _        _
| __ )| ____| |      / \
|  _ \|  _| | |     / _ \
| |_) | |___| |___ / ___ \
|____/|_____|_____/_/   \_\
http://bela.io
*/
/**
\example Audio/FFT-phase-vocoder/render.cpp

Phase Vocoder
-------------

This sketch shows an implementation of a phase vocoder and builds on the previous FFT example.
Again it uses the NE10 library, included at the top of the file.

Read the documentation on the NE10 library [here](http://projectne10.github.io/Ne10/doc/annotated.html).
*/

#include <Bela.h>
#include <libraries/ne10/NE10.h> // NEON FFT library
#include "SampleData.h"
#include "MFCCTransform.h"
#include "ArmNNFrontend.h"
#include "Log.h"

#include <cmath>
#include <algorithm>
#include <iostream>
// #define BUFFER_SIZE 8192
#define BUFFER_SIZE 8192

int gAudioChannelNum; // number of audio channels to iterate over

std::vector<float> gBigBuffer;
int gBigBufferPointer = 0;

// -----------------------------------------------
// These variables used internally in the example:

// Sample info
SampleData gSampleData; // User defined structure to get complex data from main
int gReadPtr = 0;       // Position of last read sample from file

// Auxiliary task for calculating FFT
AuxiliaryTask gFFTTask;

void process_fft_background(void *);

float gDryWet = 1;          // mix between the unprocessed and processed sound
float gPlaybackLive = 0.5f; // mix between the file playback and the live audio input
float gGain = 300;          // overall gain
std::vector<float> gInputAudio;

// ******************************

float SAMP_FREQ = 16000;
int MFCC_WINDOW_LEN = 512;    // 640;
int MFCC_WINDOW_STRIDE = 256; // 320;
int NUM_MFCC_FEATS = 10;
int NUM_MFCC_VECTORS = 49;
float MEL_LO_FREQ = 20;
float MEL_HI_FREQ = 4000;
int NUM_FBANK_BIN = 40;

std::vector<float> gMfccInput;
std::vector<int8_t> gMfccOutput;
std::vector<int8_t> gMfccOutputVector;
std::vector<int8_t> gInferenceResults;

int gMfccVectorCounter = 0;

std::unique_ptr<MFCCTransform> gMfccTransform;

int gMfccVectorSize = 49;
int debugInterval = 100;
int debugCounter = 0;
std::unique_ptr<ArmNNFrontend> nn;

// Labels for this model
std::map<int, std::string> labels =
    {
        {0, "silence"},
        {1, "unknown"},
        {2, "yes"},
        {3, "no"},
        {4, "up"},
        {5, "down"},
        {6, "left"},
        {7, "right"},
        {8, "on"},
        {9, "off"},
        {10, "stop"},
        {11, "go"}};

// ******************************
std::pair<int, float> decodeOutput(std::vector<int8_t> &modelOutput, float quantisationScale, float quantisationOffset)
{

    std::vector<float> dequantisedOutput;
    // Normalise vector values into new vector
    for (auto &value : modelOutput)
    {
        float normalisedModelOutput = quantisationScale * (static_cast<float>(value) -
                                                           static_cast<float>(quantisationOffset));
        dequantisedOutput.push_back(normalisedModelOutput);
    }

    // Get largest value in modelOutput
    const std::vector<float>::iterator &maxElementIterator = std::max_element(dequantisedOutput.begin(),
                                                                              dequantisedOutput.end());
    // Find the labelMapIndex of the largest value which corresponds to a key in a label map
    int labelMapIndex = static_cast<int>(std::distance(dequantisedOutput.begin(), maxElementIterator));

    // Round to two DP
    float maxModelOutputProbability = std::roundf((*maxElementIterator) * 100) / 100;

    return std::make_pair(labelMapIndex, maxModelOutputProbability);
}

bool setup(BelaContext *context, void *userData)
{
    // If the amout of audio input and output channels is not the same
    // we will use the minimum between input and output
    gAudioChannelNum = std::min(context->audioInChannels, context->audioOutChannels);

    // Check that we have the same number of inputs and outputs.
    if (context->audioInChannels != context->audioOutChannels)
    {
        printf("Different number of audio outputs and inputs available. Using %d channels.\n", gAudioChannelNum);
    }

    // Retrieve a parameter passed in from the initAudio() call
    gSampleData = *(SampleData *)userData;

    gBigBuffer = std::vector<float>(BUFFER_SIZE, 0.0f);
    // Allocate buffer to mirror and modify the input
    gInputAudio = std::vector<float>(context->audioFrames, 0.0f);

    //************************************

    // MfccParams mfccParams(SAMP_FREQ,
    //                         NUM_FBANK_BIN,
    //                         MEL_LO_FREQ,
    //                         MEL_HI_FREQ,
    //                         NUM_MFCC_FEATS,
    //                         MFCC_WINDOW_LEN, false,
    //                         NUM_MFCC_VECTORS);
    gMfccInput.resize(MFCC_WINDOW_LEN, 0.0f);
    gMfccOutput.resize(NUM_MFCC_FEATS, 0.0f);
    gInferenceResults.resize(12, 0);

    nn = std::make_unique<ArmNNFrontend>();
    bool ret = nn->load("ds_cnn_l_quantized.tflite");

    gMfccTransform = std::make_unique<MFCCTransform>(MFCC_WINDOW_LEN,
                                                     MFCC_WINDOW_LEN,
                                                     SAMP_FREQ,
                                                     NUM_FBANK_BIN,
                                                     MEL_LO_FREQ,
                                                     MEL_HI_FREQ,
                                                     NUM_MFCC_FEATS,
                                                     true,
                                                     nn->getQuantizationScale(),
                                                     nn->getQuantizationOffset());

    //************************************

    // Initialise auxiliary tasks
    if ((gFFTTask = Bela_createAuxiliaryTask(&process_fft_background, 90, "fft-calculation")) == 0)
        return false;

    return true;
}

// This function handles the FFT processing in this example once the buffer has
// been assembled.
void process_fft(float *inBuffer)
{
    // How many windows fit in the buffer
    auto windows = BUFFER_SIZE / MFCC_WINDOW_STRIDE - 1;

    for (int stride = 0; stride < windows; stride++)
    {
        auto hop = stride * MFCC_WINDOW_STRIDE;

        auto hopStart = inBuffer + hop;
        auto mfccInput = std::vector<float>(hopStart, hopStart + MFCC_WINDOW_LEN);
        gMfccTransform->process(mfccInput, gMfccOutput);

        // accumulate until mfcc vector is full
        if (gMfccVectorCounter < gMfccVectorSize)
        {
            std::copy(std::begin(gMfccOutput), std::end(gMfccOutput),
                      std::back_inserter(gMfccOutputVector));

            gMfccVectorCounter++;
        }
        else
        {
            nn->process(gMfccOutputVector, gInferenceResults);
            auto modelOutputDecoded = decodeOutput(gInferenceResults,
                                                nn->getQuantizationScale(),
                                                nn->getQuantizationOffset());
            std::string output = labels[modelOutputDecoded.first];

            rt_printf("-------------------------\n");
            rt_printf("Keyword \"%s\", index %d:, probability %f\n",
                    output.c_str(),
                    modelOutputDecoded.first,
                    modelOutputDecoded.second);

            gMfccVectorCounter = 0;
            gMfccOutputVector.clear();
        }
    }
}

// Function to process the FFT in a thread at lower priority
void process_fft_background(void *)
{
    process_fft(gBigBuffer.data());
}

void render(BelaContext *context, void *userData)
{
    int numAudioFrames = context->audioFrames;
    // Prep the "input" to be the sound file played in a loop
    for (int n = 0; n < numAudioFrames; n++)
    {
        gInputAudio[n] = gSampleData.samples[gReadPtr];
        if (++gReadPtr > gSampleData.sampleLen)
        {
            gReadPtr = 0;
        }
    }
    // -------------------------------------------------------------------

    for (int n = 0; n < numAudioFrames; n++)
    {
        // Count samples until we have enough for buffer
        gBigBuffer[gBigBufferPointer] = gInputAudio[n];
        gBigBufferPointer++;

        if (gBigBufferPointer > BUFFER_SIZE)
        {
            gBigBufferPointer = 0;
            Bela_scheduleAuxiliaryTask(gFFTTask);
        }

        // Copy output buffer to output
        for (int channel = 0; channel < gAudioChannelNum; channel++)
        {
            audioWrite(context, n, channel, gInputAudio[n]);
        }
    }
}

void cleanup(BelaContext *context, void *userData)
{
}
