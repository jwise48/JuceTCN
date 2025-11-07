/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin processor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <torch/script.h>
#include <torch/torch.h>
#include <string>

using namespace torch::indexing;

//==============================================================================
JuceTCNAudioProcessor::JuceTCNAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::mono(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::mono(), true)
                     #endif
                       ),
#endif
    parameters (*this, nullptr, juce::Identifier ("ronn"),
    {
        std::make_unique<juce::AudioParameterFloat> (
            juce::ParameterID("inputGain", 1),
            "Input Gain",
            juce::NormalisableRange<float>(-24.0f, 24.0f),
            0.0f),
        std::make_unique<juce::AudioParameterFloat> (
            juce::ParameterID("outputGain", 1),
            "Output Gain",
            juce::NormalisableRange<float>(-24.0f, 24.0f),
            0.0f),
        std::make_unique<juce::AudioParameterFloat> (
            juce::ParameterID("limit", 1),
            "Limit/Compress",
            juce::NormalisableRange<float>(0.0f, 1.0f),
            0.0f),
        std::make_unique<juce::AudioParameterFloat> (
            juce::ParameterID("peakReduction", 1),
            "Peak Reduction",
            juce::NormalisableRange<float>(0.0f, 100.0f),
            50.0f),
        std::make_unique<juce::AudioParameterBool> (
            juce::ParameterID("linkGain", 1),
            "Link Gain",
            false),
    })
{

    inputGainParameter     = parameters.getRawParameterValue ("inputGain");
    outputGainParameter    = parameters.getRawParameterValue ("outputGain");
    limitParameter         = parameters.getRawParameterValue ("limit");
    peakReductionParameter = parameters.getRawParameterValue ("peakReduction");

    // neural network model
    buildModel();
}

JuceTCNAudioProcessor::~JuceTCNAudioProcessor()
{
    // we may need to delete the model here
}

//==============================================================================
const juce::String JuceTCNAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool JuceTCNAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool JuceTCNAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool JuceTCNAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double JuceTCNAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int JuceTCNAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int JuceTCNAudioProcessor::getCurrentProgram()
{
    return 0;
}

void JuceTCNAudioProcessor::setCurrentProgram (int index)
{
}

const juce::String JuceTCNAudioProcessor::getProgramName (int index)
{
    return {};
}

void JuceTCNAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
}

//==============================================================================
void JuceTCNAudioProcessor::prepareToPlay (double sampleRate_, int samplesPerBlock_)
{
    // store the sample rate for future calculations
    sampleRate = sampleRate_;
    blockSamples = samplesPerBlock_;

    // setup high pass filter model
    double freq = 10.0;
    double q = 10.0;
    for (int channel = 0; channel < getTotalNumOutputChannels(); ++channel) {
        juce::IIRFilter filter;
        filter.setCoefficients(juce::IIRCoefficients::makeHighPass (sampleRate_, freq, q));
        highPassFilters.push_back(filter);
    }

    calculateReceptiveField();      // compute the receptive field, make sure it's up to date
    setupBuffers();                 // setup the buffer for handling context
}

void JuceTCNAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

#ifndef JucePlugin_PreferredChannelConfigurations
bool JuceTCNAudioProcessor::isBusesLayoutSupported (const juce::AudioProcessor::BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}
#endif

void JuceTCNAudioProcessor::calculateReceptiveField()
{
    /*
    Current Model configuration
    {"name" : "uTCN-100-C",
      "model_type" : "tcn",
      "nblocks" : 4,
      "dilation_growth" : 10,
      "kernel_size" : 5,
      "causal" : True,
      "train_fraction" : 1.00
     },
    */

    int k = 5; 
    int d = 10;
    int l = 4;
    double rf =  k * d;

    for (int layer = 1; layer < l; ++layer) {
        rf = rf + ((k-1) * pow(d,layer));
    }
    
    std::cout << "==== calculateReceptiveField ====" << std::endl;
    std::cout << "rf " << rf << std::endl;

    receptiveFieldSamples = rf; // Should always be aligned to the receptive field needed by the model.
}

void JuceTCNAudioProcessor::setupBuffers()
{
    // compute the size of the buffer which will be passed to model
    // this model is causal and therefore requires only current and past samples
    // procbuflength is based on the model architecture which determines the needed receptive field
    membuflength = (int)(receptiveFieldSamples - 1);
    procbuflength = (int)(receptiveFieldSamples - 1 + blockSamples);

    std::cout << "membuflength " << membuflength << std::endl;
    std::cout << "procbuflength " << procbuflength << std::endl;

    // Initialize the to n channels
    nInputs = getTotalNumInputChannels();

    // and membuflength samples per channel
    membuf.setSize(1, membuflength);
    membuf.clear();

    procbuf.setSize(1, procbuflength);
    procbuf.clear();
}

void JuceTCNAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    juce::ScopedNoDenormals noDenormals;
    auto inChannels  = getTotalNumInputChannels();
    auto outChannels = getTotalNumOutputChannels();
    auto numSamples = buffer.getNumSamples();

    // Early exit if no input channels
    if (inChannels == 0)
        return;

    // Clear any output channels that don't contain input data
    for (auto i = inChannels; i < outChannels; ++i)
        buffer.clear (i, 0, numSamples);
    
    // cook variables
    inputGainLn = juce::Decibels::decibelsToGain(inputGainParameter->load());
    outputGainLn = juce::Decibels::decibelsToGain(outputGainParameter->load());
    
    // we have to handle some buffer business first (this is somewhat inefficient)
    // 1. first we construct the process buffer which is [membuf, buffer]
    procbuf.copyFrom(0,0,membuf,0,0,membuflength);              // first copy the past samples into the process buffer
    procbuf.copyFrom(0,membuflength,buffer,0,0,numSamples);   // second copy the current buffer samples at the end

    // 2. now we update membuf to reflect the last N samples in proccess buffer
    membuf.copyFrom(0,0,procbuf,0,procbuflength-membuflength,membuflength);

    // 3. now move the process buffer to a tensor
    std::vector<int64_t> sizes = {procbuflength*inChannels};// size of the process buffer data
    auto* procbufptr = procbuf.getWritePointer(0);          // get pointer of the first channel
    at::Tensor frame = torch::from_blob(procbufptr, sizes); // load data from buffer into tensor type

    // frame = torch::mul(frame, inputGainLn);                 // apply the input gain first
    frame = torch::reshape(frame, {1,1,procbuflength});     // reshape so we have a batch and channel dimension

    at::Tensor conditioningParameters = torch::empty({2});
    conditioningParameters.index_put_({0}, (float)*limitParameter);
    conditioningParameters.index_put_({1}, (float)*peakReductionParameter/100.0);
    conditioningParameters = torch::reshape(conditioningParameters, {outChannels,1,2});      // reshape so we have a batch and channel dimension

    std::vector<torch::jit::IValue> inputs;                 // create special holder for model inputs
    inputs.push_back(frame);                                // add the process buffer
    inputs.push_back(conditioningParameters);               // add the parameter values (conditioning)

    at::Tensor output = model.forward(inputs).toTensor();

    // now load the output channels back into the buffer
    for (int channel = 0; channel < outChannels; ++channel) {
        auto outputData = output.index({channel,0,torch::indexing::Slice()});      // index the proper output channel
        auto outputDataPtr = outputData.data_ptr<float>();
        buffer.copyFrom(channel,0,outputDataPtr,numSamples);    // copy output data to buffer
        // remove the DC bias
        highPassFilters[channel].processSamples(buffer.getWritePointer (channel), buffer.getNumSamples());
    }
    buffer.applyGain(outputGainLn);                                  // apply the output gain

}

//==============================================================================
bool JuceTCNAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* JuceTCNAudioProcessor::createEditor()
{
    return new JuceTCNAudioProcessorEditor (*this, parameters);
}

//==============================================================================
void JuceTCNAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    juce::AudioProcessor::copyXmlToBinary (*xml, destData);
}

void JuceTCNAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState (juce::AudioProcessor::getXmlFromBinary (data, sizeInBytes));

    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xmlState));
}

//==============================================================================

void JuceTCNAudioProcessor::buildModel()
{
    try {
        // TODO: Make this configurable - for now using a placeholder path
        // Original path was: "/Users/cjstein/Code/micro-tcn/models/traced_1-uTCN-300__causal__4-10-13__fraction-0.01-bs32.pt"
        std::string model_dir = "/Users/wisjhn/Plugins/AIBookRepo/JuceTCN/model/";
        std::string model_paths[1] = {
            model_dir + "traced_1-uTCN-100-C__causal__4-10-5__fraction-1.0-bs384.pt",
        };
        model = torch::jit::load(model_paths[0]);

        // set model to evaluation mode (disables dropout, batch norm, etc.)
        model.eval();
        
        std::cout << "Model loaded and set to evaluation mode" << std::endl;
    }
    catch (const c10::Error& e) {
        std::cerr << "PyTorch error loading model: " << e.what() << std::endl;
        std::cerr << "Model will not be available for audio processing." << std::endl;
        // Model remains uninitialized - processBlock will check for this
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        std::cerr << "Model will not be available for audio processing." << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error loading model." << std::endl;
        std::cerr << "Model will not be available for audio processing." << std::endl;
    }
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new JuceTCNAudioProcessor();
}
