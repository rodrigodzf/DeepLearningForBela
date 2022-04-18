/**
 *  This is heavily based on https://github.com/ARM-software/ML-KWS-for-MCU/blob/master/Deployment/Source/MFCC/mfcc.cpp
    and https://github.com/ARM-software/armnn/tree/branches/armnn_22_02/samples/KeywordSpotting
*/

#include "MFCCTransform.h"
#include <cmath>
#include <cfloat>
#include <iostream>

MFCCTransform::MFCCTransform(unsigned int windowSize,
                             unsigned int fftSize,
                             float sampleRate,
                             float numFbankBins,
                             float melLoFreq,
                             float melHiFreq,
                             unsigned int numMfccFeatures,
                             bool useHtkMethod,
                             float quantScale,
                             float quantOffset)
{
    mFrameLen = windowSize;
    mFftSize = fftSize;
    mSampleRate = sampleRate;
    mNumFbankBins = numFbankBins;
    mMelLoFreq = melLoFreq;
    mMelHiFreq = melHiFreq;
    mUseHtkMethod = useHtkMethod;
    mNumMfccFeatures = numMfccFeatures;
    
    //**********************************
    mFft = std::make_unique<Fft>(mFftSize);
    
    // Quantized output
    mMfccOut = std::vector<int8_t>(mNumMfccFeatures, 0);
    mMelEnergies = std::vector<float>(mNumFbankBins, 0.0);
    mWindowFunc = std::vector<float>(mFrameLen);
    mBuffer = std::vector<float>(mFrameLen, 0.0);
    mFrame = std::vector<float>(mFrameLen, 0.0);

    mMelFilterBank = createMelFilterBank();
    mDctMatrix = createDCTMatrix(mNumFbankBins, 
                                  mNumMfccFeatures);
    mFilterBankInitialised = true;

    // Create Hann window function
    const auto multiplier = static_cast<float>(2 * M_PI / mFrameLen);
    for (int n = 0; n < mFrameLen; n++) 
    {
        mWindowFunc[n] = (0.5 - (0.5 * cosf(static_cast<float>(n) * multiplier)));
    }
 
    mQuantScale = quantScale;
    mQuantOffset = quantOffset;

    str();
}

MFCCTransform::~MFCCTransform()
{
}

void MFCCTransform::process(const std::vector<float>& inputData,
                            std::vector<int8_t>& outputData)
{
    // Apply window. Should equal inputData lenght!!
    for(int i = 0; i < mFrameLen; i++) 
    {
        mFrame[i] = inputData[i] * mWindowFunc[i];
    }

    // Calculate fft (2-sided)
    mFft->fft(mFrame);

    // Apply filter bank and get mel-fft
    applyMelFilterBank();

    // Multiply with DCT
    DCTQuant();

    // Copy to output
    std::copy(mMfccOut.begin(), mMfccOut.end(), outputData.begin());
}

// ********************
std::vector<float> MFCCTransform::createDCTMatrix(
                            const int32_t inputLength,
                            const int32_t coefficientCount)
{
    std::vector<float> dctMatrix(inputLength * coefficientCount);

    const float normalizer = sqrtf(2.0f/inputLength);
    const float angleIncr = M_PI/inputLength;
    float angle = 0;

    for (int32_t k = 0, m = 0; k < coefficientCount; k++, m += inputLength) 
    {
        for (int32_t n = 0; n < inputLength; n++) 
        {
            dctMatrix[m + n] = normalizer * cosf((n + 0.5f) * angle);
        }
        angle += angleIncr;
    }

    return dctMatrix;
}

void MFCCTransform::DCTQuant()
{
    float minVal = std::numeric_limits<int8_t>::min();
    float maxVal = std::numeric_limits<int8_t>::max();

    /* Take DCT. Uses matrix mul. */
    for (size_t i = 0, j = 0; i < mMfccOut.size(); ++i, j += mNumFbankBins)
    {
        float sum = 0;
        for (size_t k = 0; k < mNumFbankBins; ++k)
        {
            sum += mDctMatrix[j + k] * mMelEnergies[k];
        }
        /* Quantize to T. */
        sum = std::round((sum / mQuantScale) + mQuantOffset);
        mMfccOut[i] = static_cast<int8_t>(std::min<float>(std::max<float>(sum, minVal), maxVal));
    }
}

float MFCCTransform::DotProductF32(const float* srcPtrA, float* srcPtrB, const int srcLen)
{
    float output = 0.f;

    for (int i = 0; i < srcLen; ++i)
    {
        output += *srcPtrA++ * *srcPtrB++;
    }
    return output;
}


float MFCCTransform::melScale(const float freq, const bool useHTKMethod)
{
    if (useHTKMethod) 
    {
        return 1127.0f * logf (1.0f + freq / 700.0f);
    } 
    else 
    {
        /* Slaney formula for mel scale. */
        float mel = freq / ms_freqStep;

        if (freq >= ms_minLogHz) 
        {
            mel = ms_minLogMel + logf(freq / ms_minLogHz) / ms_logStep;
        }
        return mel;
    }
}

std::vector<std::vector<float>> MFCCTransform::createMelFilterBank()
{
    size_t numFftBins = mFrameLen / 2;
    float fftBinWidth = static_cast<float>(mSampleRate) / mFrameLen;
    
    float melLowFreq = melScale(mMelLoFreq, mUseHtkMethod);
    float melHighFreq = melScale(mMelHiFreq, mUseHtkMethod);

    float melFreqDelta = (melHighFreq - melLowFreq) / (mNumFbankBins + 1);

    std::vector<float> thisBin = std::vector<float>(numFftBins);
    std::vector<std::vector<float>> melFilterBank(mNumFbankBins);
    mFilterBankFilterFirst = std::vector<uint32_t>(mNumFbankBins);
    mFilterBankFilterLast = std::vector<uint32_t>(mNumFbankBins);

    for (size_t bin = 0; bin < mNumFbankBins; bin++) 
    {
        float leftMel = melLowFreq + bin * melFreqDelta;
        float centerMel = melLowFreq + (bin + 1) * melFreqDelta;
        float rightMel = melLowFreq + (bin + 2) * melFreqDelta;

        uint32_t firstIndex = 0;
        uint32_t lastIndex = 0;
        bool firstIndexFound = false;
        const float normaliser = 1.f;

        for (size_t i = 0; i < numFftBins; i++) 
        {
            float freq = (fftBinWidth * i);  /* Center freq of this fft bin. */
            float mel = melScale(freq, true);
            thisBin[i] = 0.0;

            if (mel > leftMel && mel < rightMel) 
            {
                float weight;
                if (mel <= centerMel) 
                {
                    weight = (mel - leftMel) / (centerMel - leftMel);
                } 
                else 
                {
                    weight = (rightMel - mel) / (rightMel - centerMel);
                }

                thisBin[i] = weight * normaliser;
                if (!firstIndexFound) 
                {
                    firstIndex = i;
                    firstIndexFound = true;
                }
                lastIndex = i;
            }
        }

        mFilterBankFilterFirst[bin] = firstIndex;
        mFilterBankFilterLast[bin] = lastIndex;

        /* Copy the part we care about. */
        for (uint32_t i = firstIndex; i <= lastIndex; i++) 
        {
            melFilterBank[bin].push_back(thisBin[i]);
        }
    }

    return melFilterBank;
}

bool MFCCTransform::applyMelFilterBank()
{
    const size_t numBanks = mMelEnergies.size();

    if (numBanks != mFilterBankFilterFirst.size() ||
        numBanks != mFilterBankFilterLast.size()) 
    {
        printf("unexpected filter bank lengths\n");
        return false;
    }

    for (size_t bin = 0; bin < numBanks; ++bin) 
    {
        auto filterBankIter = mMelFilterBank[bin].begin();
        auto end = mMelFilterBank[bin].end();
        float melEnergy = FLT_MIN;  /* Avoid log of zero at later stages */
        const uint32_t firstIndex = mFilterBankFilterFirst[bin];
        const uint32_t lastIndex = std::min<uint32_t>(mFilterBankFilterLast[bin], mFftSize - 1);

        for (uint32_t i = firstIndex; i <= lastIndex && filterBankIter != end; i++) 
        {
            float energyRep = mFft->fda(i);
            melEnergy += (*filterBankIter++ * energyRep);
        }

        // *! TODO change dynamically
        // Convert to log scale
        mMelEnergies[bin] = logf(melEnergy);
    }

    return true;
}