/**
    \ingroup keyword
    \file    MFCCTransform
    \brief   This file contains the implementation for class MFCCTransform.
             This is heavily based on https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Source/MFCC/mfcc.cpp
             and https://github.com/ARM-software/armnn/tree/branches/armnn_22_02/samples/KeywordSpotting
    \author  rodrigodzf@gmail.com
    \date    2022-04-18
*/

#pragma once

#include <vector>
#include <memory>
#include <cstdio>
#include <string>

#include <libraries/Fft/Fft.h>

class MFCCTransform
{
private:

    float ms_logStep = /*logf(6.4)*/ 1.8562979903656 / 27.0;
    float ms_freqStep = 200.0 / 3;
    float ms_minLogHz = 1000.0;
    float ms_minLogMel = ms_minLogHz / ms_freqStep;

    unsigned int mFrameLen;
    unsigned int mFftSize;
    float mSampleRate;
    unsigned int mNumFbankBins;
    float mMelLoFreq;
    float mMelHiFreq;
    bool mUseHtkMethod;
    unsigned int mNumMfccFeatures;
    float mQuantScale;
    float mQuantOffset;

    std::vector<float> mFrame;
    std::vector<float> mBuffer;
    std::vector<float> mMelEnergies;
    std::vector<float> mWindowFunc;
    std::vector<std::vector<float>> mMelFilterBank;
    std::vector<float> mDctMatrix;
    std::vector<uint32_t> mFilterBankFilterFirst;
    std::vector<uint32_t> mFilterBankFilterLast;
    bool mFilterBankInitialised;

    std::vector<int8_t> mMfccOut;
    std::unique_ptr<Fft> mFft;

public:
    void str()
    {
        printf("\n   \
                \n\t Sampling frequency:         %f\
                \n\t Number of filter banks:     %u\
                \n\t Mel frequency limit (low):  %f\
                \n\t Mel frequency limit (high): %f\
                \n\t Number of MFCC features:    %u\
                \n\t Frame length:               %u\
                \n\t Using HTK for Mel scale:    %s\
                \n\t Quatization Scale:          %f\
                \n\t Quatization Offset:         %f\n",
               mSampleRate, mNumFbankBins, mMelLoFreq,
               mMelHiFreq, mNumMfccFeatures, mFrameLen,
               mUseHtkMethod ? "yes" : "no",
               mQuantScale, mQuantOffset);
    }

    //****************************************

public:
    MFCCTransform(unsigned int windowSize,
                  unsigned int fftSize,
                  float sampleRate,
                  float numFbankBins,
                  float melLoFreq,
                  float melHiFreq,
                  unsigned int numMfccFeatures,
                  bool useHtkMethod = true,
                  float quantScale = 1.0f,
                  float quantOffset = 0.0f);
    ~MFCCTransform();

    void process(const std::vector<float> &inputData,
                 std::vector<int8_t> &outputData);

    std::vector<std::vector<float>> createMelFilterBank();
    bool applyMelFilterBank();
    float melScale(const float freq, const bool useHTKMethod);
    void DCTQuant();
    float DotProductF32(const float *srcPtrA, float *srcPtrB, const int srcLen);
    std::vector<float> createDCTMatrix(const int32_t inputLength,
                                       const int32_t coefficientCount);
};
