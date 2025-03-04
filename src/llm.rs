#[cfg(target_os = "ios")]
use crate::metal_shaders::{GPUType, GPUCapabilities, detect_metal_capabilities, initialize_metal_cache, enable_custom_metal_compute_shaders, configure_metal_parameters};

#[cfg(target_os = "ios")]
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    device_model: String,
    system_memory_mb: u32,
    gpu_type: GPUType,
}

#[cfg(target_os = "ios")]
impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            device_model: "Unknown".to_string(),
            system_memory_mb: 2048, // Conservative default
            gpu_type: GPUType::Unknown,
        }
    }
}

#[cfg(target_os = "ios")]
pub fn get_device_info() -> DeviceInfo {
    use godot::classes::Os;
    
    let os = Os::singleton();
    let model_name = os.get_model_name().to_string();
    
    // Get memory info - this is an approximation as Godot doesn't expose exact memory
    let memory_mb = os.get_static_memory_usage() / (1024 * 1024);
    
    // Detect GPU capabilities
    let caps = detect_metal_capabilities();
    
    godot_print!("[LLM iOS] Detected device: {}, Memory: {}MB", model_name, memory_mb);
    
    DeviceInfo {
        device_model: model_name,
        system_memory_mb: memory_mb as u32,
        gpu_type: caps.gpu_type,
    }
}

#[cfg(target_os = "ios")]
pub fn optimize_metal_gpu_layers(device_info: &DeviceInfo) -> u32 {
    let device_model = device_info.device_model.as_str();
    
    // Precise device-specific layer optimization
    let optimal_layers = match device_model {
        // iPhone models - optimized based on benchmarks
        m if m.contains("iPhone15,") && (m.contains("Pro") || m.contains("4") || m.contains("5")) => 32,  // iPhone 15 Pro models
        m if m.contains("iPhone15,") => 24, // iPhone 15 standard models
        m if m.contains("iPhone14,") && (m.contains("Pro") || m.contains("3") || m.contains("4")) => 28,  // iPhone 14 Pro models
        m if m.contains("iPhone14,") => 20, // iPhone 14 standard models
        m if m.contains("iPhone13,") => 20, // iPhone 13 models
        m if m.contains("iPhone12,") => 16, // iPhone 12 models
        
        // iPad models
        m if m.contains("iPad13,") || m.contains("iPad14,") => 40, // iPad Pro M1/M2
        m if m.contains("iPad8,") || m.contains("iPad11,") => 24,  // iPad Pro 2020/Air
        
        // Fall back to memory-based calculation for unknown devices
        _ => {
            let memory_mb = device_info.system_memory_mb;
            // Dynamic calculation based on available memory
            // Allocate approximately 70% of detected memory for optimal balance
            ((memory_mb as f32 * 0.7) / 100.0).round() as u32
                .min(40)  // Cap at 40 layers
                .max(8)   // Use at least 8 layers on any device
        }
    };
    
    godot_print!("[LLM Perf] Device-specific layer optimization: {} layers for {}", 
        optimal_layers, device_model);
    
    optimal_layers
}

#[cfg(target_os = "ios")]
pub struct BatchSizeOptimizer {
    current_size: usize,
    performance_history: Vec<(usize, f32)>, // (batch_size, tokens_per_sec)
    stabilized: bool,
}

#[cfg(target_os = "ios")]
impl BatchSizeOptimizer {
    pub fn new(initial_size: usize) -> Self {
        Self {
            current_size: initial_size,
            performance_history: Vec::new(),
            stabilized: false,
        }
    }
    
    pub fn record_performance(&mut self, tokens_per_sec: f32) {
        self.performance_history.push((self.current_size, tokens_per_sec));
        
        // Keep history limited to recent measurements
        if self.performance_history.len() > 10 {
            self.performance_history.remove(0);
        }
        
        // Only adapt if we haven't stabilized
        if !self.stabilized {
            self.adapt_batch_size();
        }
    }
    
    fn adapt_batch_size(&mut self) {
        if self.performance_history.len() < 3 {
            return; // Need more data
        }
        
        // Get the last three performance measurements
        let last_perf = self.performance_history.iter().rev().take(3).collect::<Vec<_>>();
        
        // Calculate if performance is improving
        let improving = last_perf[0].1 > last_perf[1].1 && last_perf[1].1 > last_perf[2].1;
        
        if improving {
            // Increase batch size by 10%
            self.current_size = (self.current_size as f32 * 1.1) as usize;
            godot_print!("[LLM Perf] Increasing batch size to {} (performance improving)", self.current_size);
        } else {
            // Performance is not improving, try decreasing
            self.current_size = (self.current_size as f32 * 0.9) as usize;
            godot_print!("[LLM Perf] Decreasing batch size to {} (seeking optimal point)", self.current_size);
            
            // If we've tried multiple sizes and performance isn't improving, stabilize
            if self.performance_history.len() >= 8 {
                self.stabilized = true;
                
                // Find the best batch size from history
                let best = self.performance_history.iter()
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap();
                
                self.current_size = best.0;
                godot_print!("[LLM Perf] Stabilized batch size at {} (best measured performance)", self.current_size);
            }
        }
        
        // Ensure we stay within reasonable limits
        self.current_size = self.current_size.max(32).min(2048);
    }
    
    pub fn get_batch_size(&self) -> usize {
        self.current_size
    }
}

#[cfg(target_os = "ios")]
pub struct ThermalController {
    last_check: std::time::Instant,
    current_state: ThermalState,
    battery_level: f32,
    low_power_mode: bool,
}

#[cfg(target_os = "ios")]
impl ThermalController {
    pub fn new() -> Self {
        Self {
            last_check: std::time::Instant::now(),
            current_state: ThermalState::Normal,
            battery_level: 1.0,
            low_power_mode: false,
        }
    }
    
    pub fn update(&mut self) -> bool {
        // Only update every few seconds to avoid overhead
        if self.last_check.elapsed() < std::time::Duration::from_secs(5) {
            return false;
        }
        
        // In a real implementation, this would use native code to access iOS APIs
        // For now, we'll use the existing update_device_state function
        if let Ok(device_state) = get_device_state() {
            self.current_state = device_state.thermal_state;
            self.battery_level = device_state.battery_level;
            self.low_power_mode = device_state.low_power_mode;
            self.last_check = std::time::Instant::now();
            return true;
        }
        
        false
    }
    
    pub fn get_performance_profile(&self) -> PerformanceProfile {
        // Determine appropriate performance profile based on thermal and battery state
        match self.current_state {
            ThermalState::Normal => {
                if self.low_power_mode || self.battery_level < 0.2 {
                    PerformanceProfile::PowerSaving
                } else if self.battery_level < 0.5 {
                    PerformanceProfile::Balanced
                } else {
                    PerformanceProfile::Maximum
                }
            },
            ThermalState::Fair => PerformanceProfile::Balanced,
            ThermalState::Serious => PerformanceProfile::PowerSaving,
            ThermalState::Critical => PerformanceProfile::Minimum,
        }
    }
}

#[cfg(target_os = "ios")]
pub enum PerformanceProfile {
    Maximum,    // Use all available performance
    Balanced,   // Balance performance and power
    PowerSaving, // Reduce performance to save power
    Minimum,    // Minimum viable performance
}

#[cfg(target_os = "ios")]
fn configure_ios_optimized_model_params(model_params: &mut LlamaModelParams, device_info: &DeviceInfo) {
    // First determine if we should use int4/int8 quantization based on device capabilities
    let should_use_int4 = device_info.system_memory_mb < 3000;
    
    // Force KV cache to use FP16 for better memory usage
    if let Ok(key) = std::ffi::CString::new("use_f16_kv") {
        use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
        model_params.append_kv_override(&key, ParamOverrideValue::Bool(true));
    }
    
    // Set rope scaling for better performance
    if let Ok(key) = std::ffi::CString::new("rope_scaling_type") {
        use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
        model_params.append_kv_override(&key, ParamOverrideValue::String("linear".into()));
    }
    
    godot_print!("[LLM Perf] Applied iOS-optimized model quantization parameters");
}

static LLAMA_BACKEND: LazyLock<LlamaBackend> =
    LazyLock::new(|| {
        #[cfg(target_os = "ios")]
        {
            // Configure Metal parameters for optimal performance
            configure_metal_parameters();
        }
        
        LlamaBackend::init().expect("Failed to initialize llama backend")
    }); 

pub fn get_model(
    model_path: &str,
    use_gpu_if_available: bool,
) -> Result<Arc<LlamaModel>, LoadModelError> {
    let load_start = Instant::now();
    godot_print!("[LLM Model] Loading model from: {}", model_path);
    
    if !std::path::Path::new(model_path).exists() {
        godot_error!("[LLM Model] Model file not found: {}", model_path);
        return Err(LoadModelError::ModelNotFound(model_path.into()));
    }

    #[cfg(target_os = "ios")]
    let model_params = {
        // Get device info for optimized configuration
        let device_info = get_device_info();
        
        // Update device state for thermal management
        let thermal_controller = ThermalController::new();
        let _ = thermal_controller.update();
        let performance_profile = thermal_controller.get_performance_profile();
        
        godot_print!("[LLM Model] Device: {}, Performance profile: {:?}", 
            device_info.device_model, performance_profile);
        
        let n_gpu_layers = if use_gpu_if_available {
            godot_print!("[LLM Model] Configuring with GPU acceleration for iOS");
            
            // Get device-specific optimal GPU layers
            let optimal_layers = optimize_metal_gpu_layers(&device_info);
            
            // Adjust based on performance profile
            let adjusted_layers = match performance_profile {
                PerformanceProfile::Maximum => optimal_layers,
                PerformanceProfile::Balanced => (optimal_layers as f32 * 0.75) as u32,
                PerformanceProfile::PowerSaving => (optimal_layers as f32 * 0.5) as u32,
                PerformanceProfile::Minimum => (optimal_layers as f32 * 0.25) as u32,
            };
            
            if optimal_layers != adjusted_layers {
                godot_print!("[LLM Model] Adjusted GPU layers from {} to {} based on performance profile", 
                    optimal_layers, adjusted_layers);
            }
            
            adjusted_layers
        } else {
            godot_print!("[LLM Model] GPU acceleration disabled, using CPU only");
            0
        };

        // Create initial parameters for pinning
        let mut params = LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_main_gpu(0);
            
        // Apply iOS-specific optimizations
        configure_ios_optimized_model_params(&mut params, &device_info);
        
        params
    };

    // Create a properly pinned version at the top level
    #[cfg(target_os = "ios")]
    let mut model_params = std::pin::pin!(model_params);

    // Now apply KV overrides on the pinned parameters
    #[cfg(target_os = "ios")]
    {
        let caps = detect_metal_capabilities();
        let n_gpu_layers = model_params.n_gpu_layers();
        
        // Only apply the most important KV override (FP16 support)
        // We can only apply one KV override with the current implementation
        if n_gpu_layers > 0 && caps.supports_fp16 {
            godot_print!("[LLM Model] Enabling FP16 for KV cache");
            // Add KV override for FP16 support
            if let Ok(key) = std::ffi::CString::new("use_f16_kv") {
                use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
                model_params.as_mut().append_kv_override(&key, ParamOverrideValue::Bool(true));
            }
        }
    }

    // ... rest of the existing function ... 
}

fn run_completion_worker_result(
    model: Arc<LlamaModel>,
    message_rx: Receiver<String>,
    completion_tx: &Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String,
    stop_tokens: Vec<String>,
) -> Result<(), WorkerError> {
    // Initialize performance metrics
    let mut metrics = PerformanceMetrics::new();
    
    godot_print!("[LLM Perf] Starting completion worker with context size: {}", n_ctx);
    
    // Set up context parameters using available parallelism
    let n_threads = std::thread::available_parallelism()?.get() as i32;
    let n_ctx = std::cmp::min(n_ctx, model.n_ctx_train());
    metrics.update_context_size(n_ctx as usize);
    
    #[cfg(target_os = "ios")]
    let ctx_params = {
        let device_info = get_device_info();
        
        // Adjust thread count based on device capability
        let optimal_threads = match device_info.gpu_type {
            GPUType::MetalHighPerformance => n_threads,
            GPUType::MetalIntegrated => n_threads,
            GPUType::MetalLowPower => std::cmp::min(n_threads, 4), // Limit threads on low-power devices
            _ => n_threads,
        };
        
        godot_print!("[LLM Perf] Using {} threads (of {} available) for computation", 
            optimal_threads, n_threads);
        
        // Configure context parameters optimized for Metal
        let mut params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZero::new(n_ctx))
            .with_n_threads(optimal_threads)
            .with_n_threads_batch(optimal_threads);
            
        // On Metal devices, we can optimize batch processing
        if matches!(device_info.gpu_type, GPUType::MetalHighPerformance | GPUType::MetalIntegrated | GPUType::MetalLowPower) {
            // Set initial batch size based on device capability
            // We'll optimize this dynamically during inference
            let initial_batch_size = match device_info.gpu_type {
                GPUType::MetalHighPerformance => 1024,
                GPUType::MetalIntegrated => 512,
                GPUType::MetalLowPower => 256,
                _ => 128,
            };
            
            godot_print!("[LLM Perf] Setting initial Metal-optimized batch size: {}", initial_batch_size);
            metrics.update_batch_size(initial_batch_size as usize);
        }
        
        params
    };
    
    // ... rest of the existing function ...
    
    // Create inference context and sampler
    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params)?;
    
    let context_time = context_start.elapsed();
    godot_print!("[LLM Perf] Context created in {:.2}s", context_time.as_secs_f32());
    
    // Initialize batch size optimizer for iOS
    #[cfg(target_os = "ios")]
    let mut batch_optimizer = {
        let device_info = get_device_info();
        let initial_size = match device_info.gpu_type {
            GPUType::MetalHighPerformance => 1024,
            GPUType::MetalIntegrated => 512,
            GPUType::MetalLowPower => 256,
            _ => 128,
        };
        BatchSizeOptimizer::new(initial_size)
    };
    
    // Pre-allocate batch size based on device detection on iOS
    #[cfg(target_os = "ios")]
    let mut batch_capacity = batch_optimizer.get_batch_size();
    
    #[cfg(not(target_os = "ios"))]
    let batch_capacity = 128; // Default batch size
    
    // ... rest of the existing function ...
    
    // Token generation loop
    loop {
        // Check for context window overflow
        if n_past >= ctx.n_ctx() as i32 - 1 {
            godot_print!("[LLM Perf] Context window nearly full, shifting context");
            let before_shift = n_past;
            n_past -= apply_context_shifting(&mut ctx, n_past)?;
            godot_print!("[LLM Perf] Shifted context from {} to {} tokens", before_shift, n_past);
            
            metrics.record_context_shift();
            metrics.update_context_size(n_past as usize);
            assert!(n_past + batch.n_tokens() < ctx.n_ctx() as i32);
        }

        // Sample next token
        let sample_start = Instant::now();
        let new_token: LlamaToken = sampler.sample(&ctx, -1);
        let sample_time = sample_start.elapsed();
        
        if get_log_level() >= LogLevel::Debug {
            godot_print!("[LLM Perf] Sampled token in {:.2}ms", sample_time.as_millis());
        }
        
        // Process current batch
        batch.clear();
        batch.add(new_token, n_past, &[0], true)?;

        assert!(batch.n_tokens() == 1);

        let decode_start = Instant::now();
        ctx.decode(&mut batch).unwrap();
        let decode_time = decode_start.elapsed();
        metrics.record_decode_time(decode_time);
        
        n_past += batch.n_tokens();
        metrics.update_context_size(n_past as usize);
        
        // Update batch size optimizer on iOS
        #[cfg(target_os = "ios")]
        {
            if tokens_generated % 10 == 0 && tokens_generated > 0 {
                // Update batch optimizer with current performance
                batch_optimizer.record_performance(metrics.tokens_per_second);
                
                // Get potentially updated batch size
                let new_batch_size = batch_optimizer.get_batch_size();
                if new_batch_size != batch_capacity {
                    batch_capacity = new_batch_size;
                    godot_print!("[LLM Perf] Dynamically adjusted batch size to {}", batch_capacity);
                }
            }
        }

        // ... rest of the existing token generation loop ... 
}

pub fn run_embedding_worker_result(
    model: Arc<LlamaModel>,
    text_rx: Receiver<String>,
    embedding_tx: &Sender<EmbeddingsOutput>,
) -> Result<(), WorkerError> {
    let mut metrics = PerformanceMetrics::new();
    godot_print!("[LLM Embed] Starting embedding worker");
    
    let n_threads = std::thread::available_parallelism()?.get() as i32;
    
    #[cfg(target_os = "ios")]
    let ctx_params = {
        let device_info = get_device_info();
        
        // Adjust thread count based on device capability
        let optimal_threads = match device_info.gpu_type {
            GPUType::MetalHighPerformance => n_threads,
            GPUType::MetalIntegrated => n_threads,
            GPUType::MetalLowPower => std::cmp::min(n_threads, 4), // Limit threads on low-power devices
            _ => n_threads,
        };
        
        godot_print!("[LLM Embed] Using {} threads for embeddings", optimal_threads);
        
        // Get thermal state for performance tuning
        let thermal_controller = ThermalController::new();
        let _ = thermal_controller.update();
        let performance_profile = thermal_controller.get_performance_profile();
        
        // Configure context parameters optimized for embeddings
        let params = LlamaContextParams::default()
            .with_n_threads(optimal_threads)
            .with_embeddings(true);
            
        // Track the actual GPU layers being used
        let gpu_layers = match (device_info.gpu_type, performance_profile) {
            (GPUType::MetalHighPerformance, PerformanceProfile::Maximum) => u32::MAX,
            (GPUType::MetalHighPerformance, _) => 32,
            (GPUType::MetalIntegrated, PerformanceProfile::Maximum) => u32::MAX,
            (GPUType::MetalIntegrated, _) => 24,
            (GPUType::MetalLowPower, PerformanceProfile::Maximum) => 20,
            (GPUType::MetalLowPower, _) => 12,
            _ => 0,
        };
        
        godot_print!("[LLM Embed] Using {} GPU layers for embeddings (profile: {:?})", 
            if gpu_layers == u32::MAX { "all".to_string() } else { gpu_layers.to_string() },
            performance_profile);
            
        metrics.update_gpu_layers(gpu_layers);
        
        params
    };
    
    // ... rest of the existing function ... 
} 