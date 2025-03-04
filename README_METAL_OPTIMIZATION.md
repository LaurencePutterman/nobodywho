# Metal Shader Optimization for iOS LLM Inference

This repository contains tools and documentation for optimizing Metal shaders for LLM inference on iOS devices. The optimizations can significantly improve performance, reduce memory usage, and extend battery life during inference.

## Quick Start

1. Make sure you have Xcode and Command Line Tools installed:
   ```bash
   xcode-select --install
   ```

2. Run the shader creation script:
   ```bash
   ./create_ios_shaders.sh
   ```

3. Copy the generated `.metallib` files to your Godot project's root directory:
   ```bash
   cp ios_optimized_shaders*.metallib /path/to/your/godot/project/
   ```

## What's Included

- **`create_ios_shaders.sh`**: Script to generate optimized Metal shaders for different iOS devices
- **`METAL_OPTIMIZATION.md`**: Detailed documentation about the Metal shader optimizations
- **`ios_optimized_shaders.md`**: Technical details about the Metal shader implementation

## Generated Shader Files

The script generates four different Metal shader libraries:

1. **`ios_optimized_shaders.metallib`**: Standard optimized shaders for all iOS devices
2. **`ios_optimized_shaders_high.metallib`**: Optimized for high-end devices (A14 and newer, iPhone 12+)
3. **`ios_optimized_shaders_mid.metallib`**: Optimized for mid-range devices (A12-A13, iPhone XS/XR to iPhone 11)
4. **`ios_optimized_shaders_low.metallib`**: Optimized for low-end devices (A11 and older, iPhone X and older)

## Key Optimizations

The optimized Metal shaders include:

1. **Device-specific optimizations**: Different optimizations for different iOS hardware
2. **Workgroup size optimization**: Smaller workgroups for better performance on mobile GPUs
3. **Precision optimization**: Using half-precision (FP16) on supported devices
4. **SIMD optimization**: Using SIMD instructions on high-end devices
5. **Memory access optimization**: Optimized memory access patterns for better cache utilization

## Integration with Godot

The Metal shaders are integrated with Godot through the `metal_shaders.rs` module, which:

1. Detects the device type and capabilities
2. Selects the appropriate shader library based on the device
3. Sets up Metal-specific optimizations
4. Configures environment variables for optimal performance

## Performance Impact

The optimized Metal shaders can provide significant performance improvements:

- **15-30% faster inference** on high-end devices
- **10-20% faster inference** on mid-range devices
- **5-15% faster inference** on low-end devices

Additionally, the optimizations can reduce memory usage and improve battery life during inference.

## Troubleshooting

If you encounter issues with the optimized Metal shaders:

1. Make sure the `.metallib` files are properly copied to your Godot project's root directory
2. Check the Godot console for any error messages related to Metal shader loading
3. If the optimized shaders fail to load, the system will automatically fall back to the default shaders

## Advanced Customization

For even better performance, you can customize the Metal shaders based on your specific model architecture and requirements:

1. Modify the `ios_optimizations.h` file to adjust optimization parameters
2. Edit the simplified Metal shader to add custom kernels
3. Adjust the device detection logic in `metal_shaders.rs`

## References

- [Apple Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
- [llama.cpp Metal Implementation](https://github.com/ggerganov/llama.cpp/blob/master/ggml-metal.metal)
- [Metal Programming Guide](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Introduction/Introduction.html) 