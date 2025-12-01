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
    }),
    setupCounter("Setup", 500, juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("tcn_setup.txt")),
    bufferManagementCounter("BufferMgmt", 500, juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("tcn_buffers.txt")),
    tensorOpsCounter("TensorOps", 500, juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("tcn_tensor.txt")),
    modelInferenceCounter("ModelInference", 500, juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("tcn_inference.txt")),
    outputProcessingCounter("OutputProc", 500, juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("tcn_output.txt")),
    totalProcessBlockCounter("TotalProcessBlock", 500, juce::File::getSpecialLocation(juce::File::tempDirectory).getChildFile("tcn_total.txt"))
{

    // initialize model configurations
    // name, type, nblocks, dilation_factor, kernel, causal
    modelConfigs = {
        {"uTCN-3-5-C", "tcn", 3, 10, 5, true},  // ran well, curious how well it emulates LA2A -rf = 490 + 512? (1002)= 11.1ms or 22.7ms
        {"uTCN-3-13-C","tcn", 3, 10, 13, true},  // ran well, could tell more effect is added over longer time windows -rf = 1450 or 32.8ms
        {"uTCN-3-17-C","tcn", 3, 10, 17, true},  // no artifacts, couldn't tell how much the effect was actually engaging though -rf = 1930 or 43.7ms
        {"uTCN-3-21-C","tcn", 3, 10, 21, true},  // no artifacts, couldn't tell how much the effect was actually engaging though -rf = 2410 or 54.6ms
        {"uTCN-3-25-C","tcn", 3, 10, 25, true},  // no artifacts, could direct the effect more, but not the transient sculpting I wanted. -rf = 2890 or 65.5ms
        {"uTCN-3-33-C","tcn", 3, 10, 33, true},  // no artifacts, could direct the effect more, but not the transient sculpting I wanted. -rf = 3850 or 87.3ms
        {"uTCN-3-37-C","tcn", 3, 10, 37, true},  // no artifacts, could direct the effect more, with some sculpting I wanted. -rf = 4330 CPU: 62% - 66% or 98.1ms
        {"uTCN-100-C", "tcn", 4, 10, 5, true},  // ran well, no artfacts -rf = 4490 or 101 ms
        {"uTCN-107-C", "tcn", 4, 10, 7, true},  // ran well, no noticable artifacts -rf = 6730 or 152ms
        {"uTCN-109-C", "tcn", 4, 10, 9, true},  // some noticable artifacts, especially when manipulating parameters and very dynammic music content -rf = 8970 or 203ms
        {"uTCN-111-C", "tcn", 4, 10, 11, true},  // pop and artifacts without manipulating parameters, struggles with basic playback -rf = 11210 or 254ms
        {"uTCN-300-C", "tcn", 4, 10, 13, true}, // pops and artifacts -rf = 13450 or 305ms
        {"uTCN-1000-C","tcn", 5, 10, 5, true}, // pops and artifacts -rf = 44490 or 1008ms
    };

    inputGainParameter     = parameters.getRawParameterValue ("inputGain");
    outputGainParameter    = parameters.getRawParameterValue ("outputGain");
    limitParameter         = parameters.getRawParameterValue ("limit");
    peakReductionParameter = parameters.getRawParameterValue ("peakReduction");

    // neural network model
    buildModel();
}

JuceTCNAudioProcessor::~JuceTCNAudioProcessor()
{
    // model is stored by value so C++ RAII (Resource Acquisition Is Initialization) will destory
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

    // compute the receptive field, make sure it's up to date
    receptiveFieldSamples = modelConfigs[currentModelIndex].calculateReceptiveField();
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

void JuceTCNAudioProcessor::setupBuffers()
{
    // compute the size of the buffer which will be passed to model
    // this model is causal and therefore requires only current and past samples
    // procbuflength is based on the model architecture which determines the needed receptive field
    membuflength = (int)(receptiveFieldSamples - 1);
    procbuflength = (int)(receptiveFieldSamples - 1 + blockSamples);

    std::cout << "membuflength " << membuflength << std::endl;
    std::cout << "procbuflength " << procbuflength << std::endl;

    // initialize the to n channels
    nInputs = getTotalNumInputChannels();

    // and membuflength samples per channel
    membuf.setSize(1, membuflength);
    membuf.clear();

    procbuf.setSize(1, procbuflength);
    procbuf.clear();
}

void JuceTCNAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midiMessages)
{
    totalProcessBlockCounter.start();
    
    setupCounter.start();
    juce::ScopedNoDenormals noDenormals;
    auto inChannels  = getTotalNumInputChannels();
    auto outChannels = getTotalNumOutputChannels();
    auto numSamples = buffer.getNumSamples();

    // early exit if no input channels
    if (inChannels == 0)
    {
        totalProcessBlockCounter.stop();
        setupCounter.stop();
        return;
    }

    // clear any output channels that don't contain input data
    for (auto i = inChannels; i < outChannels; ++i)
        buffer.clear (i, 0, numSamples);
    
    // cook variables
    inputGainLn = juce::Decibels::decibelsToGain(inputGainParameter->load());
    outputGainLn = juce::Decibels::decibelsToGain(outputGainParameter->load());
    setupCounter.stop();


    // for (int channel = 0; channel < inChannels; ++channel ) {
    //     auto* channelData = buffer.getWritePointer(channel);
    //     for (int sample = 0; sample < numSamples; ++sample) {
    //         channelData[sample] *= 1;
    //     }
    // }
    // static int debugCounter = 0;
    // static int rfFill = receptiveFieldSamples/numSamples;
    // if (debugCounter % 1000 == 0) {
    //     std::cout << "\n========== PROCESS BLOCK DEBUG (call #" << debugCounter << ") ==========" << std::endl;
    //     std::cout << "numSamples: " << numSamples << std::endl;
    //     std::cout << "receptiveFieldSamples: " << std::to_string(receptiveFieldSamples) << std::endl;
    //     std::cout << "receptive fill loops: " << std::to_string(rfFill) << std::endl;
    //     std::cout << "blockSamples: " << std::to_string(blockSamples) << std::endl;
    //     std::cout << "membuflength: " << membuflength << std::endl;
    //     std::cout << "procbuflength: " << procbuflength << std::endl;
    // }

    // Buffer management phase
    bufferManagementCounter.start();
    // we have to handle some buffer business first (this is somewhat inefficient)
    // 1. first we construct the process buffer which is [membuf, buffer]
    // ToDo: Add support for causal and non-causal models
    procbuf.copyFrom(0,0,membuf,0,0,membuflength);              // first copy the past samples into the process buffer
    procbuf.copyFrom(0,membuflength,buffer,0,0,numSamples);   // second copy the current buffer samples at the end

    // if (debugCounter % 1000 == 0|| debugCounter < rfFill * 2) {
    //     std::cout << "Copied membuf[0:" << membuflength << "] -> procbuf[0:" << membuflength << "]" << std::endl;
    //     std::cout << "Copied buffer[0:" << numSamples << "] -> procbuf[" 
    //               << membuflength << ":" << (membuflength + numSamples) << "]" << std::endl;
        
    //     const float* procbufRead = procbuf.getReadPointer(0);
    //     std::cout << "procbuf first 5 samples: ";
    //     for (int i = 0; i < std::min(512, procbuf.getNumSamples()); ++i) {
    //         std::cout << procbufRead[i] << " ";
    //     }
    //     std::cout << std::endl;
        
    //     // STEP 2: Update membuf
    //     std::cout << "\n--- STEP 2: Update membuf ---" << std::endl;
    //     std::cout << "Copying procbuf[" << (procbuflength - membuflength) 
    //               << ":" << procbuflength << "] -> membuf[0:" << membuflength << "]" << std::endl;
    // }

    // // 2. now we update membuf to reflect the last N samples in proccess buffer
    membuf.copyFrom(0,0,procbuf,0,procbuflength-membuflength,membuflength);
    bufferManagementCounter.stop();

    // if (debugCounter % 1000 == 0|| debugCounter < rfFill * 2) {
    //     const float* membufRead = membuf.getReadPointer(0);
    //     std::cout << "membuf first 5 samples: ";
    //     for (int i = 0; i < std::min(512, membuf.getNumSamples()); ++i) {
    //         std::cout << membufRead[i] << " ";
    //     }
    //     std::cout << std::endl;
        
    //     // STEP 3: Convert to tensor
    //     std::cout << "\n--- STEP 3: Convert to tensor ---" << std::endl;
    // }

    // Tensor operations
    tensorOpsCounter.start();
    // // 3. now move the process buffer to a tensor
    std::vector<int64_t> sizes = {procbuflength*inChannels};// size of the process buffer data
    auto* procbufptr = procbuf.getWritePointer(0);          // get pointer of the first channel

    // if (debugCounter % 1000 == 0 || debugCounter < rfFill * 2) {
    //     std::cout << "Tensor sizes vector: [" << sizes[0] << "]" << std::endl;
    //     std::cout << "Total elements: " << (procbuflength * inChannels) << std::endl;
    //     std::cout << "procbufptr address: " << static_cast<void*>(procbufptr) << std::endl;
    //     std::cout << "procbufptr first 5 values: ";
    //     for (int i = 0; i < 512; ++i) {
    //         std::cout << procbufptr[i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    
    at::Tensor frame = torch::from_blob(procbufptr, sizes); // load data from buffer into tensor type

    // if (debugCounter % 1000 == 0 || debugCounter < rfFill * 2) {
    //     std::cout << "Tensor created from blob" << std::endl;
    //     std::cout << "Tensor shape: " << frame.sizes() << std::endl;
    //     std::cout << "Tensor dtype: " << frame.dtype() << std::endl;
    //     std::cout << "Tensor device: " << frame.device() << std::endl;
    //     std::cout << "Tensor is_contiguous: " << frame.is_contiguous() << std::endl;
    //     std::cout << "Tensor first 5 elements: ";
    //     for (int i = 0; i < std::min<int64_t>(512, frame.size(0)); ++i) {
    //         std::cout << frame[i].item<float>() << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // frame = torch::mul(frame, inputGainLn);                 // apply the input gain first
    frame = torch::reshape(frame, {1,1,procbuflength});     // reshape so we have a batch and channel dimension

    at::Tensor conditioningParameters = torch::empty({2});
    conditioningParameters.index_put_({0}, (float)*limitParameter);
    conditioningParameters.index_put_({1}, (float)*peakReductionParameter/100.0);
    conditioningParameters = torch::reshape(conditioningParameters, {outChannels,1,2});      // reshape so we have a batch and channel dimension

    std::vector<torch::jit::IValue> inputs;                 // create special holder for model inputs
    inputs.push_back(frame);                                // add the process buffer
    inputs.push_back(conditioningParameters);               // add the parameter values (conditioning)
    tensorOpsCounter.stop();

    // Model inference
    modelInferenceCounter.start();
    
    torch::NoGradGuard no_grad;  // Explicitly disable gradient computation
    at::Tensor output = model.forward(inputs).toTensor();
    
    modelInferenceCounter.stop();

    // Output processing
    outputProcessingCounter.start();
    // now load the output channels back into the buffer
    for (int channel = 0; channel < outChannels; ++channel) {
        auto outputData = output.index({channel,0,torch::indexing::Slice()});      // index the proper output channel
        auto outputDataPtr = outputData.data_ptr<float>();
        buffer.copyFrom(channel,0,outputDataPtr,numSamples);    // copy output data to buffer
        // remove the DC bias
        // highPassFilters[channel].processSamples(buffer.getWritePointer (channel), buffer.getNumSamples());
    }
    buffer.applyGain(outputGainLn);                                  // apply the output gain
    outputProcessingCounter.stop();

    totalProcessBlockCounter.stop();

    // Periodic statistics printing
    if (++debugCounter % 1000 == 0) {
        setupCounter.printStatistics();
        bufferManagementCounter.printStatistics();
        tensorOpsCounter.printStatistics();
        modelInferenceCounter.printStatistics();
        outputProcessingCounter.printStatistics();
        totalProcessBlockCounter.printStatistics();
    }
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
        std::string modelPath = modelConfigs[currentModelIndex].modelPath();
        std::cout << "building model at: " << modelPath << std::endl;
        model = torch::jit::load(modelPath);

        // set model to evaluation mode (disables dropout, batch norm, etc.)
        model.eval();

        // OPTIMIZATION: Limit threading for real-time processing
        torch::set_num_threads(1);           // Single thread for intra-op parallelism
        torch::set_num_interop_threads(1);   // Single thread for inter-op parallelism
        
        std::cout << "Model loaded and set to evaluation mode" << std::endl;
        std::cout << "PyTorch threads: intra=" << torch::get_num_threads() 
                  << ", interop=" << torch::get_num_interop_threads() << std::endl;
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
