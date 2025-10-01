# JuceTCN - Neural Compressor Plugin

A real-time neural network-based dynamic range compressor plugin built with JUCE and LibTorch, implementing efficient neural networks for analog-style compression.

## Project Overview

JuceTCN is an audio effect plugin that uses deep learning to emulate the behavior of analog dynamic range compression. Unlike traditional compressors that use fixed mathematical models, this plugin employs a Temporal Convolutional Network (TCN) trained on real analog compressor responses to provide authentic, nuanced compression characteristics.

The plugin processes audio in real-time with low latency, making it suitable for professional music production and live performance. It offers intuitive controls for compression parameters while leveraging neural networks to capture the subtle nonlinearities and frequency-dependent behaviors of analog hardware.

This project is based on the research "Efficient neural networks for real-time modeling of analog dynamic range compression" by Christian J. Steinmetz and Joshua D. Reiss.

## References

### Source Code
- **Original Implementation**: [micro-tcn](https://github.com/csteinmetz1/micro-tcn/tree/main)
- **Research Paper**: Steinmetz, C. J., & Reiss, J. D. (2022). Efficient neural networks for real-time modeling of analog dynamic range compression. arXiv preprint arXiv:2102.06200.

### BibTeX Citation
```bibtex
@misc{steinmetz2022efficientneuralnetworksrealtime,
      title={Efficient neural networks for real-time modeling of analog dynamic range compression},
      author={Christian J. Steinmetz and Joshua D. Reiss},
      year={2022},
      eprint={2102.06200},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2102.06200}
}
```

## Features

- **Neural Network Compression**: TCN-based compression modeling trained on analog hardware
- **Real-time Processing**: Low-latency audio processing suitable for live performance
- **Intuitive Controls**:
  - Input/Output gain controls with linking option
  - Compression ratio and peak reduction parameters
  - Visual feedback showing receptive field and model parameters
- **Multiple Plugin Formats**: AU, VST3, and Standalone application
- **Cross-Platform**: macOS, Windows, and Linux support
- **High-Performance**: Optimized for real-time audio processing

## Dependencies

### Required Libraries

1. **JUCE Framework** (v6.0+)
   - Cross-platform C++ framework for audio applications
   - Provides audio I/O, GUI, and plugin hosting
   - [Official Website](https://juce.com/)

2. **LibTorch** (PyTorch C++ API)
   - C++ library for neural network inference
   - Provides TCN model loading and execution
   - [Installation Guide](https://pytorch.org/cppdist/)

3. **CMake** (v3.15+)
   - Build system generator
   - [Official Website](https://cmake.org/)

### Development Environment

- **C++17** compatible compiler (Clang, GCC, MSVC)
- **Xcode** (macOS), **Visual Studio** (Windows), or **GCC** (Linux)

## Build Instructions

### Prerequisites

1. Install JUCE Framework:
   ```bash
   # Clone JUCE repository
   git clone https://github.com/juce-framework/JUCE.git
   # JUCE should be placed at ../../../JUCE relative to this project
   ```

2. Install LibTorch:
   - Download the C++ distribution from [PyTorch website](https://pytorch.org/cppdist/)
   - Extract to a known location (e.g., `/Users/username/libtorch` or `C:\libtorch`)

3. Install CMake:
   - macOS: `brew install cmake`
   - Ubuntu: `sudo apt install cmake`
   - Windows: Download from cmake.org

### Building on macOS

1. **Configure LibTorch Path**:
   ```bash
   # Edit CMakeLists.txt and update CMAKE_PREFIX_PATH
   set(CMAKE_PREFIX_PATH "/path/to/your/libtorch")
   ```

2. **Build the Plugin**:
   ```bash
   # Make build script executable
   chmod +x build.sh

   # Run build script
   ./build.sh
   ```

3. **Alternative Manual Build**:
   ```bash
   mkdir build && cd build
   cmake .. -G Xcode -DCMAKE_PREFIX_PATH=/path/to/libtorch
   cmake --build .
   ```

### Building on Windows

1. **Configure LibTorch Path**:
   - Edit `CMakeLists.txt` and set `CMAKE_PREFIX_PATH` to your LibTorch installation path

2. **Build with Visual Studio**:
   ```cmd
   mkdir build
   cd build
   cmake .. -G "Visual Studio 16 2019" -DCMAKE_PREFIX_PATH=C:\path\to\libtorch
   cmake --build . --config Release
   ```

### Building on Linux

1. **Install Dependencies**:
   ```bash
   sudo apt update
   sudo apt install build-essential libasound2-dev libjack-jackd2-dev
   ```

2. **Configure and Build**:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch
   make -j$(nproc)
   ```

### Build Options

- **Debug Build**: Add `-DCMAKE_BUILD_TYPE=Debug` to cmake command
- **Release Build**: Add `-DCMAKE_BUILD_TYPE=Release` to cmake command
- **Custom Install Location**: Add `-DCMAKE_INSTALL_PREFIX=/custom/path`

## Usage Guide

### Loading the Plugin

1. **Build the plugin** following the instructions above
2. **Install plugin formats**:
   - **AU (macOS)**: Copy to `~/Library/Audio/Plug-Ins/Components/`
   - **VST3**: Copy to appropriate system VST3 directory
   - **Standalone**: Run the executable directly

3. **Load in DAW**:
   - **Logic Pro**: Audio Units > JuceTCN
   - **Ableton Live**: Audio Effects > JuceTCN
   - **Reaper**: FX Browser > VST3 > JuceTCN
   - **Pro Tools**: Insert > AudioSuite > JuceTCN

### Basic Usage

1. **Insert on Audio Track**: Add JuceTCN to any audio track in your DAW
2. **Adjust Input Gain**: Set input gain to achieve desired compression amount
3. **Set Compression Parameters**:
   - **Limit**: Controls compression ratio (0.0 = no compression, 1.0 = limiting)
   - **Peak Reduction**: Sets maximum peak reduction in dB
4. **Fine-tune Output**: Use output gain to compensate for level changes
5. **Link Gains** (optional): Enable link button to maintain consistent loudness

### Typical Workflow

1. **Set up monitoring**: Play audio through the plugin with moderate input gain
2. **Dial in compression**: Adjust Limit and Peak Reduction for desired effect
3. **Balance levels**: Use input/output gains to achieve target loudness
4. **Fine-tune**: Listen critically and adjust parameters for musical result

## Code Architecture & Documentation

### PluginProcessor (JuceTCNAudioProcessor)

The main audio processing class responsible for neural network inference and audio I/O.

#### Key Classes and Methods

- **`JuceTCNAudioProcessor()`**: Constructor initializes parameters and loads the neural network model
- **`prepareToPlay()`**: Sets up audio processing parameters and initializes buffers
- **`processBlock()`**: Main audio processing function that:
  1. Manages circular buffers for neural network context
  2. Applies input gain and prepares tensor input
  3. Runs neural network inference
  4. Applies output gain and high-pass filtering
- **`buildModel()`**: Loads the trained PyTorch model from disk
- **`calculateReceptiveField()`**: Computes the temporal receptive field of the network
- **`setupBuffers()`**: Initializes audio buffers for processing

#### Audio Data Flow

1. **Input Audio** → Circular Buffer Management
2. **Buffer Preparation** → Tensor Conversion
3. **Neural Network** → Compression Processing
4. **Output Processing** → Gain Application + Filtering
5. **Final Output** → DAW

### PluginEditor (JuceTCNAudioProcessorEditor)

The GUI component providing user controls and visual feedback.

#### Key Components

- **Gain Controls**: Rotary sliders for input/output gain with linking option
- **Compression Parameters**: Sliders for limit and peak reduction
- **Information Display**: Shows receptive field size and model parameters
- **Visual Design**: Clean interface with proper labeling and layout

#### Layout Structure

- **Main Panel**: Compression controls (Limit, Peak Reduction)
- **Side Panel**: Gain controls and information display
- **Header**: Plugin branding and title

## Testing & Evaluation

### Basic Functionality Testing

1. **Load Plugin**: Verify plugin loads in your DAW without errors
2. **Audio Playback**: Send audio through the plugin and confirm processing
3. **Parameter Changes**: Test all controls respond appropriately
4. **Bypass Comparison**: Compare processed vs. unprocessed audio

### Performance Testing

1. **Latency Check**: Measure round-trip latency in your DAW
2. **CPU Usage**: Monitor CPU consumption during processing
3. **Buffer Size Testing**: Test with different buffer sizes (64-2048 samples)

### Audio Quality Evaluation

1. **Listening Tests**: Compare with traditional compressors
2. **Frequency Response**: Check for unwanted artifacts
3. **Dynamic Range**: Verify compression behavior across amplitude ranges

## Contributing

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/JuceTCN.git`
3. Set up development environment following build instructions
4. Create feature branch: `git checkout -b feature/your-feature`

### Code Style Guidelines

- **C++ Standard**: Use C++17 features appropriately
- **Naming**: CamelCase for classes/methods, snake_case for variables
- **Documentation**: Comment complex algorithms and neural network operations
- **Modular Includes**: Use specific JUCE module includes, avoid `<JuceHeader.h>`

### Submitting Changes

1. **Test Thoroughly**: Ensure builds succeed on target platforms
2. **Update Documentation**: Modify README for any user-facing changes
3. **Create Pull Request**: Provide clear description of changes and rationale
4. **Code Review**: Address reviewer feedback before merging

### Issue Reporting

- Use GitHub Issues for bug reports and feature requests
- Include platform, build environment, and steps to reproduce
- Attach relevant audio examples or screenshots when possible

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Build Issues

**"Cannot find LibTorch"**
- Verify `CMAKE_PREFIX_PATH` points to correct LibTorch installation
- Ensure LibTorch version matches your PyTorch training environment

**"JUCE modules not found"**
- Confirm JUCE repository is cloned to `../../../JUCE` relative to project
- Check that JUCE submodule is properly initialized

**"Plugin not recognized by DAW"**
- Verify plugin files are in correct system directories
- Restart DAW after plugin installation
- Check plugin format compatibility (AU vs VST3)

### Platform-Specific Issues

**macOS AU Issues**
- Ensure proper code signing for AU plugins
- Check Audio MIDI Setup for AU validation

**Windows VST3 Issues**
- Verify VST3 SDK compatibility
- Check Windows audio driver configuration

**Linux Build Issues**
- Install ALSA and JACK development headers
- Verify GCC version compatibility

### Performance Issues

**High CPU Usage**
- Reduce buffer size in DAW settings
- Check for model loading errors in console
- Verify neural network model is optimized for inference

**Audio Glitches**
- Increase DAW buffer size
- Check for real-time scheduling issues
- Verify model file integrity

## Acknowledgments

### Original Research
- **Christian J. Steinmetz** and **Joshua D. Reiss** for the foundational research on efficient neural networks for real-time audio processing
- **Queen Mary University of London** for supporting the original research

### Source Code
- Thanks to Christian Steinmetz for the original micro-tcn implementation
- The JUCE community for the excellent audio framework

### Contributors
- John Wise - Plugin adaptation and CMake modernization

---

For questions, issues, or contributions, please visit the [GitHub repository](https://github.com/jwise48/JuceTCN).
