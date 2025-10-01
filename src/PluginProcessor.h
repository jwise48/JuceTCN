/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#pragma once

#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_core/juce_core.h>
#include <torch/script.h>
#include <torch/torch.h>

//==============================================================================
/**
*/
class JuceTCNAudioProcessor  : public juce::AudioProcessor
{
public:
    //==============================================================================
    JuceTCNAudioProcessor();
    ~JuceTCNAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

   #ifndef JucePlugin_PreferredChannelConfigurations
    bool isBusesLayoutSupported (const juce::AudioProcessor::BusesLayout& layouts) const override;
   #endif

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    //==============================================================================
    void calculateReceptiveField();
    void setupBuffers();

    //==============================================================================
    juce::AudioParameterInt* layers;

    //==============================================================================
    void buildModel();

    int seed = 42;
    int receptiveFieldSamples = 0; // in samples
    int blockSamples = 0; // in/out samples
    double sampleRate = 0; // in Hz

    // holder for the linear gain values
    // (don't want to convert dB -> linear on audio thread)
    float inputGainLn, outputGainLn;

private:
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JuceTCNAudioProcessor)

    //==============================================================================
    juce::AudioProcessorValueTreeState parameters;

    // is this not dangerous to use floats for values that are actually ints?
    std::atomic<float>* inputGainParameter     = nullptr;
    std::atomic<float>* outputGainParameter    = nullptr;
    std::atomic<float>* limitParameter         = nullptr;
    std::atomic<float>* peakReductionParameter = nullptr;

    juce::AudioBuffer<float> membuf, procbuf; // circular buffer to store input data
    int mbr, mbw; // read and write pointers

    int nInputs;
    int membuflength; // number of samples in the memory (past samples) buffer (rf - 1)
    int procbuflength; // number of samples in the process buffer (rf + block - 1)

    std::vector<juce::IIRFilter> highPassFilters; // high pass filters for the left and right channels

    torch::jit::script::Module model;
};
