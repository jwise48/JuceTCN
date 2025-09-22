cmake_minimum_required(VERSION 3.15)
# SHOULD CHANGE FOR EACH PLUGIN
set(PLUGIN_NAME "minimal_torch")
set(PLUGIN_ID "ABT1")

set(CMAKE_PREFIX_PATH "../../../src-resources/libtorch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

project(${PLUGIN_NAME} VERSION 0.0.1)
add_subdirectory("../../../JUCE" ./JUCE)
find_package(Torch REQUIRED)

juce_add_plugin(${PLUGIN_NAME}
    COMPANY_NAME WiseAudio
    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    IS_MIDI_EFFECT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    COPY_PLUGIN_AFTER_BUILD TRUE
    PLUGIN_MANUFACURER_CODE Wise
    # SHOULD CHANGE FOR EACH PLUGIN
    PLUGIN_CODE ${PLUGIN_ID}
    FORMATS AU VST3 Standalone
    PRODUCT_NAME ${PLUGIN_NAME})

juce_generate_juce_header(${PLUGIN_NAME})

target_sources(${PLUGIN_NAME}
    PRIVATE
    src/PluginEditor.cpp
    src/PluginProcessor.cpp)

target_compile_definitions(${PLUGIN_NAME}
    PUBLIC #
        JUCE_ALSPA=1
        JUCE_DIRECTSOUND=1
        JUCE_DISABLE_CAUTIOUS_PARAMETER_ID_CHECKING=1
        JUCE_USE_OGGVORBIS=1
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0)

target_link_libraries(${PLUGIN_NAME}
    PRIVATE
        juce::juce_audio_utils
        "${TORCH_LIBRARIES}"
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

set_property(TARGET ${PLUGIN_NAME} PROPERTY CXX_STANDARD 17)
