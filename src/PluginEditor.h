/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#pragma once

#include <juce_gui_basics/juce_gui_basics.h>
#include <juce_audio_processors/juce_audio_processors.h>
#include <juce_audio_utils/juce_audio_utils.h>
#include "PluginProcessor.h"

//==============================================================================
/**
*/
class JuceTCNAudioProcessorEditor  : public juce::AudioProcessorEditor
{
public:
    enum
    {
        paramControlHeight = 40,
        paramLabelWidth    = 80,
        paramSliderWidth   = 300
    };

    JuceTCNAudioProcessorEditor (JuceTCNAudioProcessor&, juce::AudioProcessorValueTreeState&);
    ~JuceTCNAudioProcessorEditor();

    typedef juce::AudioProcessorValueTreeState::SliderAttachment SliderAttachment;
    typedef juce::AudioProcessorValueTreeState::ButtonAttachment ButtonAttachment;
    typedef juce::AudioProcessorValueTreeState::ComboBoxAttachment ComboBoxAttachment;


    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    void updateModelState();
    void updateGains(bool inputGain);

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    JuceTCNAudioProcessor& processor;
    juce::AudioProcessorValueTreeState& valueTreeState;

    // Main panel controls
    //==============================================================================

    juce::Slider limitSlider, peakReductionSlider;
    juce::Label limitLabel, peakReductionLabel;
    std::unique_ptr<SliderAttachment> limitAttachment, peakReductionAttachment;

    juce::ToggleButton linkGainButton;
    std::unique_ptr<ButtonAttachment> linkGainAttachment;

    //ComboBox dilationsComboBox, activationsComboBox, initTypeComboBox;
    //Label dilationsLabel, activationsLabel, initTypeLabel;
    //std::unique_ptr<ComboBoxAttachment> dilationsAttachment, activationsAttachment, initTypeAttachment;

    // Side panel controls
    //==============================================================================
    juce::Label inputGainLabel, outputGainLabel;
    juce::Slider inputGainSlider, outputGainSlider;
    std::unique_ptr<SliderAttachment> inputGainAttachment, outputGainAttachment;

    juce::TextEditor receptiveFieldTextEditor, seedTextEditor, parametersTextEditor;
    juce::Label receptiveFieldLabel, seedLabel, parametersLabel;
    juce::String receptiveFieldString, seedString;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (JuceTCNAudioProcessorEditor)
};
