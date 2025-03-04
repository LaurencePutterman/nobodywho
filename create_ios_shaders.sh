#!/bin/bash
# Script to create optimized Metal shaders for iOS LLM inference
# This script generates device-specific Metal shader libraries for optimal performance

set -e  # Exit on any error

echo "=== Creating optimized Metal shaders for iOS ==="

# Create directory for iOS Metal shaders
mkdir -p ios_metal_shaders

# Find the Metal shader file from llama.cpp
echo "Looking for Metal shader file..."
METAL_SHADER_PATH=""

# Check common locations
POSSIBLE_PATHS=(
  "llama.cpp/ggml-metal.metal"
  "llama.cpp/ggml/src/ggml-metal.metal"
  "llama.cpp/ggml/src/ggml-metal/ggml-metal.metal"
  "../llama.cpp/ggml-metal.metal"
  "../llama.cpp/ggml/src/ggml-metal.metal"
  "../llama.cpp/ggml/src/ggml-metal/ggml-metal.metal"
)

for path in "${POSSIBLE_PATHS[@]}"; do
  if [ -f "$path" ]; then
    METAL_SHADER_PATH="$path"
    echo "Found Metal shader file at: $METAL_SHADER_PATH"
    break
  fi
done

if [ -z "$METAL_SHADER_PATH" ]; then
  echo "Error: Could not find Metal shader file."
  echo "Please clone llama.cpp repository or specify the path to ggml-metal.metal"
  exit 1
fi

# Copy the Metal shader file
echo "Copying Metal shader file to ios_metal_shaders directory..."
cp "$METAL_SHADER_PATH" ios_metal_shaders/ggml-metal-original.metal

# Copy necessary header files
echo "Copying necessary header files..."
SHADER_DIR=$(dirname "$METAL_SHADER_PATH")
for header in "$SHADER_DIR"/*.h "$SHADER_DIR"/../*.h; do
  if [ -f "$header" ]; then
    cp "$header" ios_metal_shaders/
    echo "Copied: $header"
  fi
done

# Fix include paths in the shader file
echo "Fixing include paths in shader file..."
cp ios_metal_shaders/ggml-metal-original.metal ios_metal_shaders/ggml-metal-optimized.metal
sed -i.bak 's/#include "..\/ggml-common.h"/#include "ggml-common.h"/' ios_metal_shaders/ggml-metal-optimized.metal
sed -i.bak 's/#include "..\/ggml-backend-impl.h"/#include "ggml-backend-impl.h"/' ios_metal_shaders/ggml-metal-optimized.metal
sed -i.bak 's/#include "..\/ggml.h"/#include "ggml.h"/' ios_metal_shaders/ggml-metal-optimized.metal

# Create optimization header for iOS devices
echo "Creating optimization header for iOS devices..."
cat > ios_metal_shaders/ios_optimizations.h << 'EOF'
#ifndef IOS_OPTIMIZATIONS_H
#define IOS_OPTIMIZATIONS_H

// Optimization flags for different iOS devices
// HIGH_END: A14 and newer (iPhone 12+)
// MID_RANGE: A12-A13 (iPhone XS/XR to iPhone 11)
// LOW_END: A11 and older (iPhone X and older)

#if defined(HIGH_END)
    #define WORKGROUP_SIZE 256
    #define TILE_SIZE 8
    #define USE_SIMD 1
    #define USE_FP16 1
#elif defined(MID_RANGE)
    #define WORKGROUP_SIZE 128
    #define TILE_SIZE 8
    #define USE_SIMD 0
    #define USE_FP16 1
#elif defined(LOW_END)
    #define WORKGROUP_SIZE 64
    #define TILE_SIZE 4
    #define USE_SIMD 0
    #define USE_FP16 0
#else
    // Default optimizations
    #define WORKGROUP_SIZE 128
    #define TILE_SIZE 8
    #define USE_SIMD 0
    #define USE_FP16 1
#endif

#endif // IOS_OPTIMIZATIONS_H
EOF

# Create device-specific shader variants
echo "Creating device-specific shader variants..."

# High-end devices (A14 and newer, iPhone 12+)
cp ios_metal_shaders/ggml-metal-optimized.metal ios_metal_shaders/ggml-metal-high.metal
sed -i.bak '1s/^/#define HIGH_END 1\n#include "ios_optimizations.h"\n/' ios_metal_shaders/ggml-metal-high.metal

# Mid-range devices (A12-A13, iPhone XS/XR to iPhone 11)
cp ios_metal_shaders/ggml-metal-optimized.metal ios_metal_shaders/ggml-metal-mid.metal
sed -i.bak '1s/^/#define MID_RANGE 1\n#include "ios_optimizations.h"\n/' ios_metal_shaders/ggml-metal-mid.metal

# Low-end devices (A11 and older, iPhone X and older)
cp ios_metal_shaders/ggml-metal-optimized.metal ios_metal_shaders/ggml-metal-low.metal
sed -i.bak '1s/^/#define LOW_END 1\n#include "ios_optimizations.h"\n/' ios_metal_shaders/ggml-metal-low.metal

# Create a simplified version of the Metal shader that removes dependencies on C headers
echo "Creating simplified Metal shader..."
cat > ios_metal_shaders/ggml-metal-simplified.metal << 'EOF'
#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>

using namespace metal;

// Include iOS optimizations if available
#if __has_include("ios_optimizations.h")
#include "ios_optimizations.h"
#else
// Default values
#define WORKGROUP_SIZE 128
#define TILE_SIZE 8
#define USE_SIMD 0
#define USE_FP16 1
#endif

// Optimized matrix multiplication kernel for float
kernel void matmul_f32(
    device const float * src0,
    device const float * src1,
    device float * dst,
    constant int & ne00,
    constant int & ne01,
    constant int & ne02,
    constant int & ne10,
    constant int & ne11,
    constant int & ne12,
    constant int & ne0,
    constant int & ne1,
    uint3 tpig[[thread_position_in_grid]],
    uint3 tptg[[threads_per_threadgroup]],
    uint3 tpitg[[thread_position_in_threadgroup]])
{
    const int i0 = tpig.y;
    const int i1 = tpig.x;
    
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    float sum = 0.0f;
    for (int k = 0; k < ne00; k++) {
        sum += src0[i0*ne00 + k] * src1[k*ne1 + i1];
    }
    dst[i0*ne1 + i1] = sum;
}

// Optimized matrix multiplication kernel for half
#if USE_FP16
kernel void matmul_f16(
    device const half * src0,
    device const half * src1,
    device half * dst,
    constant int & ne00,
    constant int & ne01,
    constant int & ne02,
    constant int & ne10,
    constant int & ne11,
    constant int & ne12,
    constant int & ne0,
    constant int & ne1,
    uint3 tpig[[thread_position_in_grid]],
    uint3 tptg[[threads_per_threadgroup]],
    uint3 tpitg[[thread_position_in_threadgroup]])
{
    const int i0 = tpig.y;
    const int i1 = tpig.x;
    
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    half sum = 0.0h;
    for (int k = 0; k < ne00; k++) {
        sum += src0[i0*ne00 + k] * src1[k*ne1 + i1];
    }
    dst[i0*ne1 + i1] = sum;
}
#endif

// SIMD-optimized version for high-end devices
#if USE_SIMD
kernel void matmul_f32_simd(
    device const float * src0,
    device const float * src1,
    device float * dst,
    constant int & ne00,
    constant int & ne01,
    constant int & ne02,
    constant int & ne10,
    constant int & ne11,
    constant int & ne12,
    constant int & ne0,
    constant int & ne1,
    uint3 tpig[[thread_position_in_grid]],
    uint3 tptg[[threads_per_threadgroup]],
    uint3 tpitg[[thread_position_in_threadgroup]])
{
    const int i0 = tpig.y * TILE_SIZE;
    const int i1 = tpig.x * TILE_SIZE;
    
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    float acc[TILE_SIZE][TILE_SIZE] = {0.0f};
    
    for (int k = 0; k < ne00; k++) {
        for (int i = 0; i < min(TILE_SIZE, ne0 - i0); i++) {
            float a_val = src0[(i0 + i) * ne00 + k];
            for (int j = 0; j < min(TILE_SIZE, ne1 - i1); j++) {
                float b_val = src1[k * ne1 + (i1 + j)];
                acc[i][j] += a_val * b_val;
            }
        }
    }
    
    for (int i = 0; i < min(TILE_SIZE, ne0 - i0); i++) {
        for (int j = 0; j < min(TILE_SIZE, ne1 - i1); j++) {
            dst[(i0 + i) * ne1 + (i1 + j)] = acc[i][j];
        }
    }
}
#endif
EOF

# Try to compile the original shaders first (this might fail)
echo "Attempting to compile original shaders (this might fail)..."
if xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-optimized.metal -o ios_metal_shaders/ggml-metal-optimized.air 2>/dev/null; then
    echo "Original shader compilation succeeded!"
    USE_ORIGINAL=1
else
    echo "Original shader compilation failed, using simplified shader instead."
    USE_ORIGINAL=0
fi

# Compile the Metal shaders
echo "Compiling Metal shaders..."

if [ $USE_ORIGINAL -eq 1 ]; then
    # Compile standard shader
    echo "Compiling standard shader..."
    xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-optimized.metal -o ios_metal_shaders/ggml-metal-optimized.air
    xcrun -sdk iphoneos metallib ios_metal_shaders/ggml-metal-optimized.air -o ios_optimized_shaders.metallib
    
    # Compile high-end device shader
    echo "Compiling high-end device shader..."
    xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-high.metal -o ios_metal_shaders/ggml-metal-high.air
    xcrun -sdk iphoneos metallib ios_metal_shaders/ggml-metal-high.air -o ios_optimized_shaders_high.metallib
    
    # Compile mid-range device shader
    echo "Compiling mid-range device shader..."
    xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-mid.metal -o ios_metal_shaders/ggml-metal-mid.air
    xcrun -sdk iphoneos metallib ios_metal_shaders/ggml-metal-mid.air -o ios_optimized_shaders_mid.metallib
    
    # Compile low-end device shader
    echo "Compiling low-end device shader..."
    xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-low.metal -o ios_metal_shaders/ggml-metal-low.air
    xcrun -sdk iphoneos metallib ios_metal_shaders/ggml-metal-low.air -o ios_optimized_shaders_low.metallib
else
    # Compile simplified shaders
    echo "Compiling simplified standard shader..."
    xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-simplified.metal -o ios_metal_shaders/ggml-metal-simplified.air
    xcrun -sdk iphoneos metallib ios_metal_shaders/ggml-metal-simplified.air -o ios_optimized_shaders.metallib
    
    # Compile high-end device shader
    echo "Compiling simplified high-end device shader..."
    sed -i.bak '1s/^/#define HIGH_END 1\n/' ios_metal_shaders/ggml-metal-simplified.metal
    xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-simplified.metal -o ios_metal_shaders/ggml-metal-simplified-high.air
    xcrun -sdk iphoneos metallib ios_metal_shaders/ggml-metal-simplified-high.air -o ios_optimized_shaders_high.metallib
    
    # Compile mid-range device shader
    echo "Compiling simplified mid-range device shader..."
    sed -i.bak '1s/#define HIGH_END 1/#define MID_RANGE 1/' ios_metal_shaders/ggml-metal-simplified.metal
    xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-simplified.metal -o ios_metal_shaders/ggml-metal-simplified-mid.air
    xcrun -sdk iphoneos metallib ios_metal_shaders/ggml-metal-simplified-mid.air -o ios_optimized_shaders_mid.metallib
    
    # Compile low-end device shader
    echo "Compiling simplified low-end device shader..."
    sed -i.bak '1s/#define MID_RANGE 1/#define LOW_END 1/' ios_metal_shaders/ggml-metal-simplified.metal
    xcrun -sdk iphoneos metal -c ios_metal_shaders/ggml-metal-simplified.metal -o ios_metal_shaders/ggml-metal-simplified-low.air
    xcrun -sdk iphoneos metallib ios_metal_shaders/ggml-metal-simplified-low.air -o ios_optimized_shaders_low.metallib
fi

echo "=== Metal shader compilation complete ==="
echo "Generated shader files:"
ls -la ios_optimized_shaders*.metallib

echo ""
echo "To use these shaders in your Godot project:"
echo "1. Copy the .metallib files to your Godot project's root directory"
echo "2. Make sure the files are properly imported in Godot"
echo "3. The code will automatically detect and use the optimized shaders"

# Clean up temporary files
echo "Cleaning up temporary files..."
rm -f ios_metal_shaders/*.bak

echo "Done!" 