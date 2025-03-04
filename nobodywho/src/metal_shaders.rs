use godot::prelude::*;
use std::env;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GPUType {
    Unknown,
    MetalHighPerformance,  // A14 and newer (iPhone 12 and newer)
    MetalIntegrated,       // A12-A13 (iPhone XS/XR to iPhone 11)
    MetalLowPower,         // A11 and older (iPhone X and older)
    Other,
}

#[derive(Debug, Clone)]
pub struct GPUCapabilities {
    pub gpu_type: GPUType,
    pub supports_fp16: bool,
    pub supports_simd_group: bool,
    pub max_threads_per_group: u32,
}

impl Default for GPUCapabilities {
    fn default() -> Self {
        Self {
            gpu_type: GPUType::Unknown,
            supports_fp16: false,
            supports_simd_group: false,
            max_threads_per_group: 512,
        }
    }
}

/// Detects Metal capabilities of the current device
pub fn detect_metal_capabilities() -> GPUCapabilities {
    use godot::classes::Os;
    
    let os = Os::singleton();
    let model_name = os.get_model_name().to_string();
    
    // Determine GPU type based on device model
    let gpu_type = match model_name.as_str() {
        // High-end devices (A14 and newer)
        m if m.contains("iPhone15,") || m.contains("iPhone14,") || m.contains("iPhone13,") => GPUType::MetalHighPerformance,
        m if m.contains("iPad13,") || m.contains("iPad14,") => GPUType::MetalHighPerformance, // iPad Pro M1/M2
        
        // Mid-range devices (A12-A13)
        m if m.contains("iPhone12,") || m.contains("iPhone11,") => GPUType::MetalIntegrated,
        m if m.contains("iPad8,") || m.contains("iPad11,") => GPUType::MetalIntegrated, // iPad Pro 2020/Air
        
        // Low-power devices (A11 and older)
        m if m.contains("iPhone10,") || m.contains("iPhone9,") => GPUType::MetalLowPower,
        
        // Unknown or other devices
        _ => GPUType::Unknown,
    };
    
    // Determine capabilities based on GPU type
    let supports_fp16 = match gpu_type {
        GPUType::MetalHighPerformance => true,
        GPUType::MetalIntegrated => true,
        GPUType::MetalLowPower => false,
        _ => false,
    };
    
    let supports_simd_group = match gpu_type {
        GPUType::MetalHighPerformance => true,
        _ => false,
    };
    
    let max_threads_per_group = match gpu_type {
        GPUType::MetalHighPerformance => 1024,
        GPUType::MetalIntegrated => 512,
        GPUType::MetalLowPower => 256,
        _ => 512,
    };
    
    godot_print!("[Metal] Detected GPU type: {:?}", gpu_type);
    godot_print!("[Metal] FP16 support: {}", supports_fp16);
    godot_print!("[Metal] SIMD group support: {}", supports_simd_group);
    
    GPUCapabilities {
        gpu_type,
        supports_fp16,
        supports_simd_group,
        max_threads_per_group,
    }
}

/// Initializes Metal cache for better performance
pub fn initialize_metal_cache() -> bool {
    // Set Metal cache directory to a writable location
    let user_data_dir = godot::classes::Os::singleton().get_user_data_dir();
    let cache_dir = user_data_dir.to_string();
    
    let metal_cache_dir = format!("{}/metal_cache", cache_dir);
    
    // Create directory if it doesn't exist
    if !std::path::Path::new(&metal_cache_dir).exists() {
        if let Err(e) = std::fs::create_dir_all(&metal_cache_dir) {
            godot_error!("[Metal] Failed to create Metal cache directory: {}", e);
            return false;
        }
    }
    
    // Set Metal cache environment variable
    env::set_var("GGML_METAL_CACHE_DIR", &metal_cache_dir);
    godot_print!("[Metal] Initialized Metal cache at: {}", metal_cache_dir);
    true
}

/// Enables custom Metal compute shaders for optimized performance
pub fn enable_custom_metal_compute_shaders() -> bool {
    // Check if we have optimized shader files available
    let mut shader_path = "res://ios_optimized_shaders.metallib".to_string();
    let gpu_capabilities = detect_metal_capabilities();
    
    // Select the appropriate shader based on GPU capabilities
    if gpu_capabilities.gpu_type == GPUType::MetalHighPerformance {
        if godot::classes::FileAccess::file_exists(&GString::from("res://ios_optimized_shaders_high.metallib")) {
            shader_path = "res://ios_optimized_shaders_high.metallib".to_string();
            godot_print!("[Metal] Using high-end optimized Metal shaders");
        }
    } else if gpu_capabilities.gpu_type == GPUType::MetalIntegrated {
        if godot::classes::FileAccess::file_exists(&GString::from("res://ios_optimized_shaders_mid.metallib")) {
            shader_path = "res://ios_optimized_shaders_mid.metallib".to_string();
            godot_print!("[Metal] Using mid-range optimized Metal shaders");
        }
    } else if gpu_capabilities.gpu_type == GPUType::MetalLowPower {
        if godot::classes::FileAccess::file_exists(&GString::from("res://ios_optimized_shaders_low.metallib")) {
            shader_path = "res://ios_optimized_shaders_low.metallib".to_string();
            godot_print!("[Metal] Using low-end optimized Metal shaders");
        }
    }
    
    // Check if the selected shader file exists
    if godot::classes::FileAccess::file_exists(&GString::from(&shader_path)) {
        // Set the environment variable to use our custom shaders
        env::set_var("GGML_METAL_SHADER_PATH", shader_path);
        godot_print!("[Metal] Enabled custom optimized Metal compute shaders for {:?}", gpu_capabilities.gpu_type);
        
        // Set additional Metal-specific optimizations based on device type
        match gpu_capabilities.gpu_type {
            GPUType::MetalHighPerformance => {
                env::set_var("GGML_METAL_STREAM_COMMAND_BUFFER", "1");
                env::set_var("GGML_METAL_POOL_SIZE", "16");
            },
            GPUType::MetalIntegrated => {
                env::set_var("GGML_METAL_STREAM_COMMAND_BUFFER", "1");
                env::set_var("GGML_METAL_POOL_SIZE", "8");
            },
            GPUType::MetalLowPower => {
                env::set_var("GGML_METAL_POOL_SIZE", "4");
            },
            _ => {}
        }
        
        return true;
    }
    
    godot_print!("[Metal] Using default Metal compute shaders");
    false
}

/// Configures optimal Metal parameters for the LLM based on device capabilities
pub fn configure_metal_parameters() {
    // Initialize Metal caching
    let _ = initialize_metal_cache();
    
    // Enable optimized Metal compute shaders
    let _ = enable_custom_metal_compute_shaders();
    
    // Set Metal-specific options
    env::set_var("GGML_METAL_NDEBUG", "1"); // Disable Metal debugging for performance
} 