/*
  ==============================================================================

    This file was auto-generated!

    It contains the basic framework code for a JUCE plugin editor.

  ==============================================================================
*/

#include "PluginProcessor.h"
#include "PluginEditor.h"

//==============================================================================
JuceTCNAudioProcessorEditor::JuceTCNAudioProcessorEditor (JuceTCNAudioProcessor& p, juce::AudioProcessorValueTreeState& vts)
    : juce::AudioProcessorEditor (&p), audioProcessor (p), valueTreeState (vts)
{

    getLookAndFeel().setColour (juce::Slider::thumbColourId, juce::Colours::grey);
    getLookAndFeel().setColour (juce::Slider::trackColourId, juce::Colours::lightgrey);
    getLookAndFeel().setColour (juce::Slider::backgroundColourId, juce::Colours::lightgrey);
    getLookAndFeel().setColour (juce::Slider::textBoxBackgroundColourId, juce::Colours::white);
    getLookAndFeel().setColour (juce::Slider::textBoxTextColourId, juce::Colours::darkgrey);
    getLookAndFeel().setColour (juce::Slider::textBoxHighlightColourId, juce::Colours::darkgrey);
    getLookAndFeel().setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::white);
    getLookAndFeel().setColour (juce::Slider::rotarySliderFillColourId, juce::Colours::lightgrey);
    getLookAndFeel().setColour (juce::Slider::rotarySliderOutlineColourId, juce::Colours::lightgrey);

    getLookAndFeel().setColour (juce::ComboBox::backgroundColourId, juce::Colours::white);
    getLookAndFeel().setColour (juce::ComboBox::textColourId, juce::Colours::darkgrey);
    getLookAndFeel().setColour (juce::ComboBox::outlineColourId, juce::Colours::lightgrey);
    getLookAndFeel().setColour (juce::ComboBox::arrowColourId, juce::Colours::darkgrey);

    getLookAndFeel().setColour (juce::ToggleButton::textColourId, juce::Colours::darkgrey);
    getLookAndFeel().setColour (juce::ToggleButton::tickColourId, juce::Colours::darkgrey);
    getLookAndFeel().setColour (juce::ToggleButton::tickDisabledColourId, juce::Colours::lightgrey);

    getLookAndFeel().setColour (juce::Label::textColourId, juce::Colours::darkgrey);

    getLookAndFeel().setColour (juce::PopupMenu::backgroundColourId, juce::Colours::white);
    getLookAndFeel().setColour (juce::PopupMenu::textColourId, juce::Colours::darkgrey);
    getLookAndFeel().setColour (juce::PopupMenu::highlightedBackgroundColourId, juce::Colours::lightgrey);
    getLookAndFeel().setColour (juce::PopupMenu::highlightedTextColourId, juce::Colours::darkgrey);

    linkGainButton.setButtonText ("Link");

    juce::Colour fillColour = juce::Colour (0xffececec); // side panel color

    inputGainSlider.setRange(-24, 24);
    inputGainSlider.setSliderStyle (juce::Slider::Rotary);
    inputGainSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 50, 24);//(Slider::NoTextBox, false, 0, 0);
    inputGainSlider.onValueChange = [this] {updateGains(true);};
    inputGainSlider.setValue (juce::Decibels::gainToDecibels(audioProcessor.inputGainLn));
    inputGainSlider.setColour (juce::Slider::textBoxBackgroundColourId, fillColour);
    inputGainSlider.setColour (juce::Slider::textBoxOutlineColourId, fillColour);
    inputGainLabel.setText ("Input", juce::dontSendNotification);
    inputGainLabel.attachToComponent (&inputGainSlider, true);

    outputGainSlider.setRange(-24, 24);
    outputGainSlider.setSliderStyle (juce::Slider::Rotary);
    outputGainSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 50, 24);//(Slider::NoTextBox, false, 0, 0);
    outputGainSlider.onValueChange = [this] {updateGains(false);};
    outputGainSlider.setValue (juce::Decibels::gainToDecibels(audioProcessor.outputGainLn));
    outputGainSlider.setColour (juce::Slider::textBoxBackgroundColourId, fillColour);
    outputGainSlider.setColour (juce::Slider::textBoxOutlineColourId, fillColour);
    outputGainLabel.setText ("Output", juce::dontSendNotification);
    outputGainLabel.attachToComponent (&outputGainSlider, true);

    limitSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 30, 24);
    peakReductionSlider.setTextBoxStyle (juce::Slider::TextBoxRight, false, 30, 24);

    addAndMakeVisible (limitSlider);
    addAndMakeVisible (peakReductionSlider);
    addAndMakeVisible (inputGainSlider);
    addAndMakeVisible (outputGainSlider);
    //addAndMakeVisible (dilationsComboBox);
    addAndMakeVisible (linkGainButton);
    addAndMakeVisible (inputGainLabel);
    addAndMakeVisible (outputGainLabel);

    // attach labels
    addAndMakeVisible (limitLabel);
    limitLabel.setText ("Limit", juce::dontSendNotification);
    limitLabel.attachToComponent (&limitSlider, true);
    addAndMakeVisible (peakReductionLabel);
    peakReductionLabel.setText ("Peak Reduction", juce::dontSendNotification);
    peakReductionLabel.attachToComponent (&peakReductionSlider, true);

    // add options to comboboxes
    //dilationsComboBox.addItem ("1^n", 1);
    //dilationsComboBox.addItem ("2^n", 2);
    //dilationsComboBox.addItem ("3^n", 3);
    //dilationsComboBox.addItem ("4^n", 4);

    //addAndMakeVisible (dilationsLabel);
    //dilationsLabel.setText ("dilation", dontSendNotification);
    //dilationsLabel.attachToComponent (&dilationsComboBox, true);

    receptiveFieldTextEditor.setColour (juce::TextEditor::backgroundColourId, fillColour);
    receptiveFieldTextEditor.setColour (juce::TextEditor::outlineColourId, fillColour);
    receptiveFieldTextEditor.setColour (juce::TextEditor::textColourId, juce::Colours::darkgrey);
    receptiveFieldTextEditor.setColour (juce::TextEditor::highlightColourId, juce::Colours::darkgrey);
    receptiveFieldTextEditor.setReadOnly(true);
    receptiveFieldTextEditor.setFont(juce::Font (15.0f));
    receptiveFieldTextEditor.setText(std::to_string(audioProcessor.receptiveFieldSamples), false);  // ToDo: Add actual visualization of receptive field
    receptiveFieldLabel.setText ("receptive field", juce::dontSendNotification);
    receptiveFieldLabel.attachToComponent (&receptiveFieldTextEditor, true);
    addAndMakeVisible(receptiveFieldTextEditor);
    addAndMakeVisible(receptiveFieldLabel);

    parametersTextEditor.setColour (juce::TextEditor::backgroundColourId, fillColour);
    parametersTextEditor.setColour (juce::TextEditor::outlineColourId, fillColour);
    parametersTextEditor.setColour (juce::TextEditor::textColourId, juce::Colours::darkgrey);
    parametersTextEditor.setColour (juce::TextEditor::highlightColourId, juce::Colours::darkgrey);
    parametersTextEditor.setReadOnly(true);
    parametersTextEditor.setFont(juce::Font (15.0f));
    parametersTextEditor.setText("0", false);
    parametersLabel.setText ("parameters", juce::dontSendNotification);
    parametersLabel.attachToComponent (&parametersTextEditor, true);
    addAndMakeVisible(parametersTextEditor);
    addAndMakeVisible(parametersLabel);

    /*
    seedTextEditor.setColour (TextEditor::backgroundColourId, fillColour);
    seedTextEditor.setColour (TextEditor::outlineColourId, fillColour);
    seedTextEditor.setColour (TextEditor::textColourId, Colours::darkgrey);
    seedTextEditor.setColour (TextEditor::highlightColourId, Colours::darkgrey);
    seedTextEditor.setReadOnly(false);
    seedTextEditor.setFont(Font (15.0f));
    seedTextEditor.setText("0", false);
    seedLabel.setText ("seed", dontSendNotification);
    seedLabel.attachToComponent (&seedTextEditor, true);
    addAndMakeVisible(seedTextEditor);
    addAndMakeVisible(seedLabel);
    */

    limitAttachment.reset         (new SliderAttachment   (valueTreeState, "limit", limitSlider));
    peakReductionAttachment.reset (new SliderAttachment   (valueTreeState, "peakReduction", peakReductionSlider));
    //channelsAttachment.reset    (new SliderAttachment   (valueTreeState, "channels", channelsSlider));
    inputGainAttachment.reset   (new SliderAttachment   (valueTreeState, "inputGain", inputGainSlider));
    outputGainAttachment.reset  (new SliderAttachment   (valueTreeState, "outputGain", outputGainSlider));
    //dilationsAttachment.reset   (new ComboBoxAttachment (valueTreeState, "dilation", dilationsComboBox));
    //activationsAttachment.reset (new ComboBoxAttachment (valueTreeState, "activation", activationsComboBox));
    //initTypeAttachment.reset    (new ComboBoxAttachment (valueTreeState, "initType", initTypeComboBox));
    //useBiasAttachment.reset     (new ButtonAttachment   (valueTreeState, "useBias", useBiasButton));
    linkGainAttachment.reset    (new ButtonAttachment   (valueTreeState, "linkGain", linkGainButton));
    //depthwiseAttachment.reset   (new ButtonAttachment   (valueTreeState, "depthwise", depthwiseButton));
    //seedAttachment.reset        (new TextBoxAttachment  (valueTreeState, "seed", seedTextEditor));

    // callbacks for updating the model (not all parameters)
    //layersSlider.onValueChange   = [this] { updateModelState(); };
    //kernelSlider.onValueChange   = [this] { updateModelState(); };
    //channelsSlider.onValueChange = [this] { updateModelState(); };
    //dilationsComboBox.onChange   = [this] { updateModelState(); };
    //activationsComboBox.onChange = [this] { updateModelState(); };
    //initTypeComboBox.onChange    = [this] { updateModelState(); };
    //useBiasButton.onStateChange  = [this] { updateModelState(); };
    //depthwiseButton.onStateChange = [this] { updateModelState(); };

    setSize (600, 300);
}

JuceTCNAudioProcessorEditor::~JuceTCNAudioProcessorEditor()
{
}

//==============================================================================
void JuceTCNAudioProcessorEditor::updateGains(bool inputGain)
{
  if (inputGain == true){
    audioProcessor.inputGainLn = juce::Decibels::decibelsToGain(static_cast<float>(inputGainSlider.getValue()));
    inputGainSlider.setValue (juce::Decibels::gainToDecibels(audioProcessor.inputGainLn));
    if (linkGainButton.getToggleState()) {
      float outputGaindB = -1.0f * static_cast<float>(inputGainSlider.getValue());
      audioProcessor.outputGainLn = juce::Decibels::decibelsToGain(outputGaindB);
      outputGainSlider.setValue (juce::Decibels::gainToDecibels(audioProcessor.outputGainLn));
    }
  }
  else {
    audioProcessor.outputGainLn = juce::Decibels::decibelsToGain(static_cast<float>(outputGainSlider.getValue()));
    outputGainSlider.setValue (juce::Decibels::gainToDecibels(audioProcessor.outputGainLn));
    if (linkGainButton.getToggleState()) {
      float inputGaindB = -1.0f * static_cast<float>(outputGainSlider.getValue());
      audioProcessor.inputGainLn = juce::Decibels::decibelsToGain(inputGaindB);
      inputGainSlider.setValue (juce::Decibels::gainToDecibels(audioProcessor.inputGainLn));
    }
  }
}

//==============================================================================
void JuceTCNAudioProcessorEditor::updateModelState()
{
  audioProcessor.modelConfigs[audioProcessor.currentModelIndex].calculateReceptiveField();
  float rfms = static_cast<float>(audioProcessor.receptiveFieldSamples) / static_cast<float>(audioProcessor.sampleRate) * 1000.0f;
  receptiveFieldTextEditor.setText(juce::String(rfms, 1));
  //int parameters = audioProcessor.model->getNumParameters();
  //parametersTextEditor.setText(String(parameters));
}

//==============================================================================
void JuceTCNAudioProcessorEditor::paint (juce::Graphics& g)
{
    // fill the whole window white
    g.fillAll (juce::Colours::white);

    // set the font size and draw text to the screen
    g.setFont (15.0f);

    // fill the right panel with grey
    {
      juce::Colour fillColour = juce::Colour (0xffececec);
      g.setColour (fillColour);
      g.fillRect (400, 0, 300, 300); // draw side bar on the right
      g.fillRect (0, 0, 30, 300);    // draw strip on the left

      g.setColour (juce::Colours::grey);
      g.setFont (juce::Font ("Source Sans Variable", 32.0f, juce::Font::plain).withTypefaceStyle ("Light")); //.withExtraKerningFactor (0.147f));
      g.drawText ("ncomp", 350, 0, 300, 70, juce::Justification::centred, true);
      g.setFont (juce::Font ("Source Sans Variable", 10.0f, juce::Font::plain).withTypefaceStyle ("Light")); //.withExtraKerningFactor (0.147f));
      g.drawText (audioProcessor.modelConfigs[audioProcessor.currentModelIndex].name, 350, 0, 300, 105, juce::Justification::centred, true);
      g.drawText ("TCN", 350, 0, 300, 125, juce::Justification::centred, true);
    }
}

void JuceTCNAudioProcessorEditor::resized()
{
    auto marginTop          = 32;
    auto contentPadding     = 12;
    auto sectionPadding     = 18;
    auto contentItemHeight  = 24;
    auto rotaryItemHeight   = 55;
    auto stripWidth         = 30;
    auto sidePanelWidth     = 200;

    // side panel
    auto area = getLocalBounds();

    area.removeFromTop(marginTop + 60);
    area.removeFromLeft(600 - sidePanelWidth);
    area.removeFromLeft(sectionPadding+30);
    area.removeFromRight(sectionPadding+10);

    // place the gain sliders
    inputGainSlider.setBounds  (area.removeFromTop (rotaryItemHeight));
    area.removeFromTop(6);
    outputGainSlider.setBounds (area.removeFromTop (rotaryItemHeight));

    area.removeFromTop(12);
    area.removeFromLeft(65); // slide over the textboxes
    //area.removeFromRight(); // slide over the textboxes
    receptiveFieldTextEditor.setBounds(area.removeFromTop (contentItemHeight));
    area.removeFromTop(1);
    parametersTextEditor.setBounds(area.removeFromTop (contentItemHeight));
    area.removeFromTop(1);

    // center panel
    area = getLocalBounds();

    area.removeFromTop(marginTop);
    area.removeFromLeft(stripWidth);
    area.removeFromLeft(sectionPadding+65);
    area.removeFromRight(sidePanelWidth);
    area.removeFromRight(sectionPadding);


    limitSlider.setBounds        (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    peakReductionSlider.setBounds        (area.removeFromTop (contentItemHeight));
    /*
    area.removeFromTop(contentPadding);
    channelsSlider.setBounds      (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    dilationsComboBox.setBounds   (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    activationsComboBox.setBounds (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);
    initTypeComboBox.setBounds    (area.removeFromTop (contentItemHeight));
    area.removeFromTop(contentPadding);

    auto toggleArea = area.removeFromTop (contentItemHeight);
    useBiasButton.setBounds       (toggleArea);
    linkGainButton.setBounds      (toggleArea.removeFromRight(60));
    depthwiseButton.setBounds     (toggleArea.removeFromRight(120));
    */
}
