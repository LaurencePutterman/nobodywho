// iOS-optimized Metal shaders for LLM inference
// Based on llama.cpp ggml-metal.metal

// Enable mobile optimizations
#define OPTIMIZE_FOR_MOBILE 1

// Optimized for iOS devices
#define OPTIMIZE_FOR_APPLE_GPU 1

// Use smaller workgroup sizes for mobile GPUs
#define METAL_WORKGROUP_SIZE 8

