#!/bin/bash

# Build script for JuceTCN plugin
# Update the CMAKE_PREFIX_PATH below to point to your LibTorch installation

rm -rf build
mkdir build
cd build
cmake .. -G Xcode -DCMAKE_PREFIX_PATH=/Users/wisjhn/src_resources/libtorch ..
cmake --build .
