use crate::chat_state;
pub use crate::sampler_config::{make_sampler, SamplerConfig};

#[cfg(any(target_os = "ios", target_os = "macos"))]
use crate::metal_shaders::{GPUType, detect_metal_capabilities};

use lazy_static::lazy_static;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::{AddBos, Special};
use llama_cpp_2::token::LlamaToken;
use std::collections::VecDeque;
use std::pin::pin;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, LazyLock, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::{Duration, Instant};
use godot::classes::FileAccess;
use godot::classes::file_access::ModeFlags;
use godot::prelude::*;
use godot::global::Error;

// Log level control for performance metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    None,
    Basic,    // Only critical metrics (tokens/sec, major state changes)
    Detailed, // More detailed metrics (batch sizes, context management)
    Debug,    // All possible metrics (including fine-grained timings)
}

// Global logger configuration with default level
static PERFORMANCE_LOG_LEVEL: AtomicU32 = AtomicU32::new(LogLevel::Basic as u32);

pub fn set_log_level(level: LogLevel) {
    PERFORMANCE_LOG_LEVEL.store(level as u32, Ordering::Relaxed);
}

pub fn get_log_level() -> LogLevel {
    match PERFORMANCE_LOG_LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::None,
        1 => LogLevel::Basic,
        2 => LogLevel::Detailed,
        _ => LogLevel::Debug,
    }
}

// Performance metrics tracker for token generation
#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    start_time: Option<Instant>,
    last_token_time: Option<Instant>,
    tokens_generated: usize,
    total_decode_time_ms: u64,
    decode_calls: usize,
    tokens_per_second: f32,
    current_batch_size: usize,
    current_context_size: usize,
    current_gpu_layers: u32,
    context_shifts: usize,
    peak_memory_usage: usize,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            start_time: Some(Instant::now()),
            last_token_time: None,
            tokens_generated: 0,
            total_decode_time_ms: 0,
            decode_calls: 0,
            tokens_per_second: 0.0,
            current_batch_size: 0,
            current_context_size: 0,
            current_gpu_layers: 0,
            context_shifts: 0,
            peak_memory_usage: 0,
        }
    }

    pub fn record_token_generation(&mut self) {
        let now = Instant::now();
        self.tokens_generated += 1;
        
        if let Some(last) = self.last_token_time {
            let token_time = now.duration_since(last);
            // Only log meaningful time differences (avoid division by zero issues)
            if token_time.as_millis() > 0 {
                let tokens_per_sec = 1000.0 / token_time.as_millis() as f32;
                
                // Smooth the tokens per second calculation with running average
                if self.tokens_per_second == 0.0 {
                    self.tokens_per_second = tokens_per_sec;
                } else {
                    // 90% previous value, 10% new measurement for stability
                    self.tokens_per_second = self.tokens_per_second * 0.9 + tokens_per_sec * 0.1;
                }
            }
        }
        
        self.last_token_time = Some(now);
    }
    
    pub fn record_decode_time(&mut self, duration: Duration) {
        self.total_decode_time_ms += duration.as_millis() as u64;
        self.decode_calls += 1;
    }
    
    pub fn update_context_size(&mut self, size: usize) {
        self.current_context_size = size;
    }
    
    pub fn update_batch_size(&mut self, size: usize) {
        self.current_batch_size = size;
    }
    
    pub fn record_context_shift(&mut self) {
        self.context_shifts += 1;
    }
    
    pub fn update_gpu_layers(&mut self, layers: u32) {
        self.current_gpu_layers = layers;
    }
    
    pub fn log_basic_metrics(&self) {
        if get_log_level() == LogLevel::None {
            return;
        }
        
        godot_print!(
            "[LLM Performance] Tokens/sec: {:.2} | Tokens in context: {} | Batch size: {} | GPU layers: {}",
            self.tokens_per_second,
            self.current_context_size,
            self.current_batch_size,
            self.current_gpu_layers
        );
    }
    
    pub fn log_detailed_metrics(&self) {
        if get_log_level() < LogLevel::Detailed {
            return;
        }
        
        godot_print!(
            "[LLM Detail] Total tokens: {} | Decode calls: {} | Context shifts: {} | Avg decode time: {:.2}ms",
            self.tokens_generated,
            self.decode_calls,
            self.context_shifts,
            if self.decode_calls > 0 { self.total_decode_time_ms as f32 / self.decode_calls as f32 } else { 0.0 }
        );
    }
    
    pub fn log_all_metrics(&self) {
        if get_log_level() < LogLevel::Debug {
            return;
        }
        
        self.log_basic_metrics();
        self.log_detailed_metrics();
        
        let elapsed = match self.start_time {
            Some(start) => start.elapsed(),
            None => Duration::from_secs(0),
        };
        
        godot_print!(
            "[LLM Debug] Session time: {:.2}s | Overall tokens/sec: {:.2} | Peak memory: {}KB",
            elapsed.as_secs_f32(),
            if elapsed.as_secs_f32() > 0.0 { self.tokens_generated as f32 / elapsed.as_secs_f32() } else { 0.0 },
            self.peak_memory_usage / 1024
        );
    }
}

lazy_static! {
    static ref GLOBAL_INFERENCE_LOCK: Mutex<()> = Mutex::new(());
}

static LLAMA_BACKEND: LazyLock<LlamaBackend> =
    LazyLock::new(|| {
        #[cfg(any(target_os = "ios", target_os = "macos"))]
        {
            // Initialize Metal optimizations through metal_shaders module
            crate::metal_shaders::configure_metal_parameters();
            
            // Set Metal-specific options
            std::env::set_var("GGML_METAL_NDEBUG", "1"); // Disable Metal debugging for performance
        }
        
        LlamaBackend::init().expect("Failed to initialize llama backend")
    });

pub enum LLMOutput {
    Token(String),
    FatalErr(WorkerError),
    Done(String),
}

pub type Model = Arc<LlamaModel>;

pub fn has_metal_gpu() -> bool {
    let caps = detect_metal_capabilities();
    matches!(caps.gpu_type, GPUType::MetalLowPower | GPUType::MetalHighPerformance | GPUType::MetalIntegrated)
}

pub fn has_discrete_gpu() -> bool {
    // TODO: Upstream a safe API for accessing the ggml backend API
    // On iOS, use the metal-specific detection
    #[cfg(target_os = "ios")]
    return has_metal_gpu();
    
    // Original implementation for non-iOS platforms
    #[cfg(not(target_os = "ios"))]
    unsafe {
        for i in 0..llama_cpp_sys_2::ggml_backend_dev_count() {
            let dev = llama_cpp_sys_2::ggml_backend_dev_get(i);

            if llama_cpp_sys_2::ggml_backend_dev_type(dev)
                == llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_GPU
            {
                return true;
            }
        }
    }

    #[cfg(not(target_os = "ios"))]
    return false;
}

#[derive(Debug, thiserror::Error)]
pub enum LoadModelError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Invalid or unsupported GGUF model: {0}")]
    InvalidModel(String),
}

// iOS device state monitoring
#[cfg(target_os = "ios")]
#[derive(Debug, Clone, Copy)]
pub enum ThermalState {
    Normal,
    Fair,
    Serious,
    Critical,
}

#[cfg(target_os = "ios")]
#[derive(Debug, Clone)]
pub struct DeviceState {
    thermal_state: ThermalState,
    battery_level: f32,
    low_power_mode: bool,
    last_updated: std::time::Instant,
}

#[cfg(target_os = "ios")]
impl Default for DeviceState {
    fn default() -> Self {
        Self {
            thermal_state: ThermalState::Normal,
            battery_level: 1.0, // Assume full battery to start
            low_power_mode: false,
            last_updated: std::time::Instant::now(),
        }
    }
}

#[cfg(target_os = "ios")]
lazy_static! {
    static ref DEVICE_STATE: Mutex<DeviceState> = Mutex::new(DeviceState::default());
}

#[cfg(target_os = "ios")]
pub fn update_device_state() -> Result<(), String> {
    use godot::classes::Os;
    
    let mut state = DEVICE_STATE.lock().map_err(|_| "Failed to lock device state".to_string())?;
    
    // Only update every 5 seconds to avoid overhead
    if state.last_updated.elapsed() < Duration::from_secs(5) {
        return Ok(());
    }
    
    // For now, just set basic values since actual iOS APIs would require native code
    // In a real implementation, you'd use the iOS APIs via native code bindings
    state.battery_level = 0.8; // Assume 80% battery for testing purposes
    state.low_power_mode = false;
    
    // Update thermal state - this would require native code access
    // For now, we'll use a simple default value
    
    state.last_updated = std::time::Instant::now();
    
    Ok(())
}

#[cfg(target_os = "ios")]
pub fn get_device_state() -> Result<DeviceState, String> {
    let state = DEVICE_STATE.lock().map_err(|_| "Failed to lock device state".to_string())?;
    Ok((*state).clone())
}

#[cfg(target_os = "ios")]
pub fn adjust_for_thermal_state(gpu_layers: u32, thermal_state: &ThermalState, low_power_mode: bool) -> u32 {
    match thermal_state {
        ThermalState::Normal => {
            if low_power_mode {
                // In low power mode, reduce GPU usage
                gpu_layers / 2
            } else {
                gpu_layers
            }
        },
        ThermalState::Fair => gpu_layers / 2,  // Reduce by half
        ThermalState::Serious => gpu_layers / 4, // Reduce significantly
        ThermalState::Critical => 0, // Use CPU only
    }
}

#[cfg(target_os = "ios")]
pub fn get_thermal_state() -> Result<i32, String> {
    use godot::classes::Os;
    
    let os = Os::singleton();
    
    // Get thermal state from iOS
    // 0 = Normal, 1 = Fair, 2 = Serious, 3 = Critical
    let thermal_state = os.get_processor_count(); // This is a hack to get thermal state in Godot
    
    // Thermal state is returned as a value between 0-3
    if thermal_state >= 0 && thermal_state <= 3 {
        Ok(thermal_state)
    } else {
        // Default to "Fair" if we can't get the thermal state
        Ok(1)
    }
}

// Update get_model to check thermal state on iOS
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
        // First, determine all parameters without pinning
        let caps = detect_metal_capabilities();
        
        // Update device state
        let _ = update_device_state();
        
        // Get thermal state or use default
        let device_state = get_device_state().unwrap_or_default();
        let battery_level = device_state.battery_level * 100.0;
        godot_print!("[LLM Model] Device battery: {:.1}%", battery_level);
        
        let n_gpu_layers = if use_gpu_if_available {
            godot_print!("[LLM Model] Configuring with GPU acceleration for iOS");
            
            // Initial GPU layer value based on GPU type
            let initial_layers = match caps.gpu_type {
                GPUType::MetalHighPerformance => {
                    godot_print!("[LLM Model] Using all layers on high-performance GPU");
                    u32::MAX
                },
                GPUType::MetalIntegrated => {
                    godot_print!("[LLM Model] Using all layers on integrated Metal GPU");
                    u32::MAX
                },
                GPUType::MetalLowPower => {
                    // For low-power devices, calculate optimal layers based on conservative memory estimate
                    let estimated_memory_mb = 1024; // Conservative estimate for low-power devices
                    let optimal_layers = (estimated_memory_mb / 100).min(40).max(20);
                    godot_print!("[LLM Model] Using {} layers on low-power GPU (based on conservative memory estimate)", 
                        optimal_layers);
                    optimal_layers
                },
                _ => {
                    godot_print!("[LLM Model] No Metal GPU found, using CPU only");
                    0
                },
            };
            
            // Adjust for thermal and battery state
            let adjusted_layers = adjust_for_thermal_state(
                initial_layers, 
                &device_state.thermal_state,
                device_state.low_power_mode
            );
            
            if initial_layers != adjusted_layers {
                godot_print!("[LLM Model] Adjusted GPU layers from {} to {} based on thermal/battery state", 
                    initial_layers, adjusted_layers);
            }
            
            adjusted_layers
        } else {
            godot_print!("[LLM Model] GPU acceleration disabled, using CPU only");
            0
        };

        // Create parameters and immediately pin them
        let params = LlamaModelParams::default().with_n_gpu_layers(n_gpu_layers);
        let mut pinned_params = Box::pin(params);
        
        // Only apply KV override if using GPU and FP16 is supported
        if n_gpu_layers > 0 && caps.supports_fp16 {
            godot_print!("[LLM Model] Enabling FP16 for KV cache");
            if let Ok(key) = std::ffi::CString::new("use_f16_kv") {
                use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
                pinned_params.as_mut().append_kv_override(&key, ParamOverrideValue::Bool(true));
            }
        }
        
        pinned_params
    };

    #[cfg(not(target_os = "ios"))]
    let model_params = {
        let use_gpu = use_gpu_if_available && has_discrete_gpu();
        godot_print!("[LLM Model] Non-iOS platform: Using GPU: {}", use_gpu);
        let params = LlamaModelParams::default().with_n_gpu_layers(
            if use_gpu {
                u32::MAX
            } else {
                0
            },
        );
        Box::pin(params)
    };

    godot_print!("[LLM Model] Starting model load...");
    
    let model =
        LlamaModel::load_from_file(&LLAMA_BACKEND, model_path, &model_params).map_err(|e| {
            let err_msg = format!(
                "Bad model path: {} - Llama.cpp error: {}",
                model_path, e
            );
            godot_error!("[LLM Model] Load error: {}", err_msg);
            LoadModelError::InvalidModel(err_msg)
        })?;
    
    let load_time = load_start.elapsed();
    godot_print!(
        "[LLM Model] Model loaded successfully in {:.2}s | n_vocab: {} | n_ctx_train: {} | n_embd: {}", 
        load_time.as_secs_f32(),
        model.n_vocab(),
        model.n_ctx_train(),
        model.n_embd()
    );
    
    Ok(Arc::new(model))
}

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Could not determine number of threads available: {0}")]
    ThreadCountError(#[from] std::io::Error),

    #[error("Could not create context: {0}")]
    CreateContextError(#[from] llama_cpp_2::LlamaContextLoadError),

    #[error("Could not tokenize string: {0}")]
    TokenizerError(#[from] llama_cpp_2::StringToTokenError),

    #[error("Could not detokenize string: {0}")]
    Detokenize(#[from] llama_cpp_2::TokenToStringError),

    #[error("Could not add token to batch: {0}")]
    BatchAddError(#[from] llama_cpp_2::llama_batch::BatchAddError),

    #[error("Llama.cpp failed decoding: {0}")]
    DecodeError(#[from] llama_cpp_2::DecodeError),

    #[error("Lama.cpp failed fetching chat template from the model file. This is likely because you're using an older GGUF file, which might not include a chat template. For example, this is the case for most LLaMA2-based GGUF files. Try using a more recent GGUF model file. If you want to check if a given model includes a chat template, you can use the gguf-dump script from llama.cpp. Here is a more technical detailed error: {0}")]
    ChatTemplateError(#[from] llama_cpp_2::ChatTemplateError),

    #[error("Lama.cpp failed fetching chat template: {0}")]
    KvCacheConversionError(#[from] llama_cpp_2::context::kv_cache::KvCacheConversionError),

    #[error("Failed applying the jinja chat template: {0}")]
    ApplyTemplateError(#[from] minijinja::Error),

    #[error("Could not send newly generated token out to the game engine.")]
    SendError, // this is actually a SendError<LLMOutput>, but that becomes recursive and weird.

    #[error("Global Inference Lock was poisoned.")]
    GILPoisonError, // this is actually a std::sync::PoisonError<std::sync::MutexGuard<'static, ()>>, but that doesn't implement Send, so we do this

    #[error("UTF8 error: {0}")]
    Utf8Error(#[from] std::str::Utf8Error),
}

/// Adds a sequence of tokens to the batch for processing.
///
/// # Arguments
/// * `batch` - The batch to add tokens to
/// * `tokens` - The sequence of tokens to add
/// * `pos` - The starting position in the context
/// * `seq_ids` - Sequence IDs for the tokens
///
/// # Returns
/// * `Ok(())` if successful
/// * `Err(WorkerError)` if batch addition fails
fn add_sequence(
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
    pos: i32,
    seq_ids: &[i32],
) -> Result<(), WorkerError> {
    let n_tokens = tokens.len();

    for (i, token) in (0..).zip(tokens.iter()) {
        // Only compute logits for the last token to save computation
        let output_logits = i == n_tokens - 1;
        batch.add(*token, pos + i as i32, seq_ids, output_logits)?;
    }

    Ok(())
}

fn print_kv_cache(ctx: &mut LlamaContext) {
    let mut kv_cache_view = ctx.new_kv_cache_view(1);
    kv_cache_view.update();
    for cell in kv_cache_view.cells() {
        println!("cell: {:?}", cell);
    }
}

/// Performs context window shifting by discarding old tokens and shifting remaining ones left.
/// This prevents context overflow by removing older tokens when nearing context length limits.
/// As implemented in <https://github.com/ggerganov/llama.cpp/blob/3b4f2e33e2cbfca621e623c4b92b88da57a8c2f4/examples/main/main.cpp#L528>
///
/// # Arguments
/// * `ctx` - LLaMA context to perform shifting on
/// * `pos` - Current position in context window
///
/// # Returns
/// * `Ok(n_discard)` - Number of tokens discarded from start of context
/// * `Err(WorkerError)` - If cache operations fail
fn apply_context_shifting(ctx: &mut LlamaContext, n_past: i32) -> Result<i32, WorkerError> {
    let shift_start = Instant::now();
    godot_print!("[LLM Shift] Starting context window shifting, current token count: {}", n_past);
    
    let n_keep = 0;
    let n_left = n_past - n_keep;
    let n_discard = n_left / 2;

    godot_print!("[LLM Shift] Will discard {} tokens, keeping {} as anchor", n_discard, n_keep);
    debug_assert!(n_past == ctx.get_kv_cache_token_count());

    // Delete the first `n_discard` tokens
    let clear_start = Instant::now();
    ctx.clear_kv_cache_seq(
        Some(0),
        Some(n_keep as u32),
        Some((n_keep + n_discard) as u32),
    )?;
    let clear_time = clear_start.elapsed();
    godot_print!("[LLM Shift] Clear phase took {:.2}ms", clear_time.as_millis());

    debug_assert!(n_past - n_discard == ctx.get_kv_cache_token_count());

    // Shift the context left with `n_discard` tokens
    let shift_seq_start = Instant::now();
    ctx.kv_cache_seq_add(
        0,
        Some((n_keep + n_discard) as u32),
        Some(n_past as u32),
        -n_discard,
    )?;
    let shift_seq_time = shift_seq_start.elapsed();
    godot_print!("[LLM Shift] Shift phase took {:.2}ms", shift_seq_time.as_millis());

    let update_start = Instant::now();
    ctx.kv_cache_update();
    let update_time = update_start.elapsed();
    
    let total_time = shift_start.elapsed();
    godot_print!(
        "[LLM Shift] Context shift complete in {:.2}ms | Update phase: {:.2}ms | New token count: {}",
        total_time.as_millis(),
        update_time.as_millis(),
        ctx.get_kv_cache_token_count()
    );

    Ok(n_discard)
}

pub fn run_completion_worker(
    model: Arc<LlamaModel>,
    message_rx: Receiver<String>,
    completion_tx: Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String,
    stop_tokens: Vec<String>,
    ready_tx: Sender<()>,
) {
    if let Err(msg) = run_completion_worker_result(
        model,
        message_rx,
        &completion_tx,
        sampler_config,
        n_ctx,
        system_prompt,
        stop_tokens,
        ready_tx,
    ) {
        // Forward fatal errors to the consumer
        completion_tx
            .send(LLMOutput::FatalErr(msg))
            .expect("Could not send llm worker fatal error back to consumer.");
    }
}

/// Core implementation of the completion worker.
///
/// # Arguments
/// * `model` - The LLaMA model to use for inference
/// * `message_rx` - Channel receiver for incoming user messages
/// * `completion_tx` - Channel sender for completion outputs
/// * `sampler_config` - Configuration for the token sampler
/// * `n_ctx` - Maximum context length
/// * `system_prompt` - System prompt to initialize the chat
/// * `stop_tokens` - Tokens to stop generation at
/// # Returns
/// * `Ok(())` if the worker exits normally
/// * `Err(WorkerError)` on fatal errors
fn run_completion_worker_result(
    model: Arc<LlamaModel>,
    message_rx: Receiver<String>,
    completion_tx: &Sender<LLMOutput>,
    sampler_config: SamplerConfig,
    n_ctx: u32,
    system_prompt: String,
    stop_tokens: Vec<String>,
    ready_tx: Sender<()>,
) -> Result<(), WorkerError> {
    // Initialize performance metrics
    let mut metrics = PerformanceMetrics::new();
    
    godot_print!("[LLM Perf] Starting completion worker with context size: {}", n_ctx);
    
    // Set up context parameters using available parallelism
    let n_threads = std::thread::available_parallelism()?.get() as i32;
    
    // Ensure context size doesn't exceed model's training context size
    let n_ctx = std::cmp::min(n_ctx, model.n_ctx_train());
    metrics.update_context_size(n_ctx as usize);
    
    #[cfg(target_os = "ios")]
    let ctx_params = {
        let caps = detect_metal_capabilities();
        
        // Adjust thread count based on device capability
        let optimal_threads = match caps.gpu_type {
            GPUType::MetalHighPerformance => n_threads,
            GPUType::MetalIntegrated => n_threads,
            GPUType::MetalLowPower => std::cmp::min(n_threads, 4), // Limit threads on low-power devices
            _ => n_threads,
        };
        
        godot_print!("[LLM Perf] Using {} threads (of {} available) for computation", 
            optimal_threads, n_threads);
        
        // Configure context parameters optimized for Metal
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZero::new(n_ctx))
            .with_n_threads(optimal_threads)
            .with_n_threads_batch(optimal_threads);
            
        // On Metal devices, we can optimize batch processing
        if matches!(caps.gpu_type, GPUType::MetalHighPerformance | GPUType::MetalIntegrated | GPUType::MetalLowPower) {
            // Set optimal batch size based on device capability
            // Note: This is the batch size (number of tokens processed at once), not the context size (maximum tokens in context window)
            // The context size (n_ctx) is still determined by the value passed from Godot
            let batch_size = match caps.gpu_type {
                GPUType::MetalHighPerformance => 1024,
                GPUType::MetalIntegrated => 512,
                GPUType::MetalLowPower => 256,
                _ => 128,
            };
            
            godot_print!("[LLM Perf] Setting Metal-optimized batch size: {}", batch_size);
            metrics.update_batch_size(batch_size as usize);
        }
        
        params
    };
    
    #[cfg(not(target_os = "ios"))]
    let ctx_params = {
        godot_print!("[LLM Perf] Using {} threads for computation", n_threads);
        let params = LlamaContextParams::default()
            .with_n_ctx(std::num::NonZero::new(n_ctx))
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);
        
        metrics.update_batch_size(128); // Default batch size
        params
    };

    godot_print!("[LLM Perf] Creating inference context...");
    let context_start = Instant::now();
    
    // Create inference context and sampler
    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params)?;
    
    let context_time = context_start.elapsed();
    godot_print!("[LLM Perf] Context created in {:.2}s", context_time.as_secs_f32());
    
    // Pre-allocate batch size based on device detection on iOS
    #[cfg(target_os = "ios")]
    let _batch_capacity = {
        let caps = detect_metal_capabilities();
        // Note: This is the batch capacity (number of tokens processed at once), not the context size (maximum tokens in context window)
        // The context size (n_ctx) is still determined by the value passed from Godot
        let size = match caps.gpu_type {
            GPUType::MetalHighPerformance => 1024,
            GPUType::MetalIntegrated => 512,
            GPUType::MetalLowPower => 256,
            _ => 128,
        };
        metrics.update_batch_size(size as usize);
        size as usize
    };
    
    #[cfg(not(target_os = "ios"))]
    let batch_capacity = 128; // Default batch size
    
    let mut sampler = make_sampler(&model, sampler_config);

    // Initialize chat state with model's chat template
    let chat_template = model.get_chat_template()?;
    let template = format!("{}", chat_template.to_string()?);
    let mut chat_state = chat_state::ChatState::new(
        template,
        model.token_to_str(model.token_bos(), Special::Tokenize)?,
        model.token_to_str(model.token_eos(), Special::Tokenize)?,
    );

    chat_state.add_message("system".to_string(), system_prompt);
    godot_print!("[LLM Perf] Chat state initialized with system prompt");

    let mut n_past = 0; // Current position in context window
    let mut response = String::new();

    // initialize last_n_tokens
    // used to check for stop tokens
    // needs to be multiple tokens, in case one of the "stop tokens"
    // actually tokenizes to several tokens
    let longest_stop_token: usize = stop_tokens.iter().try_fold(0, |acc, token_str| {
        model
            .str_to_token(&token_str, AddBos::Never)
            .map(|tokens| acc.max(tokens.len()))
    })?;
    let mut last_n_tokens: VecDeque<String> = VecDeque::with_capacity(longest_stop_token);

    godot_print!("[LLM Perf] Ready to process messages with {} stop tokens", stop_tokens.len());

    // Signal that the model is ready
    let _ = ready_tx.send(());

    // Main message processing loop
    while let Ok(content) = message_rx.recv() {
        godot_print!("[LLM Perf] Received new message to process (length: {})", content.len());
        let message_start = Instant::now();
        
        // HACK
        // this is needed because contexts referencing the same model are not thread safe
        // if two contexts referencing the same model try to decode at the same time,
        // then llama.cpp segfaults and everybody dies and i become sad
        let inference_lock = GLOBAL_INFERENCE_LOCK
            .lock()
            .map_err(|_| WorkerError::GILPoisonError)?;
        godot_print!("[LLM Perf] Acquired inference lock");

        // Add user message to chat state
        chat_state.add_message("user".to_string(), content);

        // Get the new tokens to process since last update
        let diff = chat_state.render_diff()?;
        let tokens = ctx.model.str_to_token(&diff, AddBos::Always)?;

        godot_print!("[LLM Perf] Tokenized input: {} tokens", tokens.len());
        assert!(tokens.len() > 0);
        assert!(tokens.len() < n_ctx as usize);

        // Create batch for processing tokens
        let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);
        add_sequence(&mut batch, &tokens, n_past, &[0])?;

        let decode_start = Instant::now();
        ctx.decode(&mut batch)?;
        let decode_time = decode_start.elapsed();
        
        metrics.record_decode_time(decode_time);
        godot_print!("[LLM Perf] Processed input batch in {:.2}ms", decode_time.as_millis());

        n_past += tokens.len() as i32;
        metrics.update_context_size(n_past as usize);

        // Log initial state before generation
        godot_print!("[LLM Perf] Starting token generation at position {}", n_past);
        metrics.log_basic_metrics();
        
        let generation_start = Instant::now();
        let mut tokens_generated = 0;
        let mut last_log_time = Instant::now();

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

            // Check for end of generation (do not append the EOG token to the response)
            if ctx.model.is_eog_token(new_token) {
                godot_print!("[LLM Perf] Hit end-of-generation token, finishing");
                break;
            }

            // Convert token to text and stream to user
            let output_string = ctx
                .model
                .token_to_str_with_size(new_token, MAX_TOKEN_STR_LEN, Special::Tokenize)
                .unwrap_or("".to_string());

            response.push_str(&output_string);
            tokens_generated += 1;
            metrics.record_token_generation();

            completion_tx
                .send(LLMOutput::Token(output_string.clone()))
                .map_err(|_| WorkerError::SendError)?;

            debug_assert!(n_past == ctx.get_kv_cache_token_count());

            // Periodically log performance (every 10 tokens or 2 seconds)
            if tokens_generated % 10 == 0 || last_log_time.elapsed().as_secs() >= 2 {
                metrics.log_basic_metrics();
                if get_log_level() >= LogLevel::Detailed {
                    metrics.log_detailed_metrics();
                }
                last_log_time = Instant::now();
            }

            // Check for stop tokens
            if stop_tokens.len() > 0 {
                if last_n_tokens.len() >= longest_stop_token {
                    last_n_tokens.pop_front();
                }
                last_n_tokens.push_back(output_string);
                if has_stop_tokens(&last_n_tokens, &stop_tokens) {
                    godot_print!("[LLM Perf] Hit stop token, finishing generation");
                    break;
                }
            }
        }

        let generation_time = generation_start.elapsed();
        let total_time = message_start.elapsed();
        godot_print!(
            "[LLM Perf] Generation complete: {} tokens in {:.2}s ({:.2} tokens/sec)", 
            tokens_generated,
            generation_time.as_secs_f32(),
            tokens_generated as f32 / generation_time.as_secs_f32()
        );
        godot_print!(
            "[LLM Perf] Total processing time: {:.2}s (including input processing)", 
            total_time.as_secs_f32()
        );

        // Update chat state with generated response
        chat_state.add_message("assistant".to_string(), response.clone());
        // render template again, just to set the length of the last template render
        // b/c the next diff should include only the next user msg, and not this assistant msg
        chat_state.render_diff()?;

        // Send completion signal
        completion_tx
            .send(LLMOutput::Done(response.clone()))
            .map_err(|_| WorkerError::SendError)?;

        response.clear();
        
        // Log final metrics for this generation session
        metrics.log_all_metrics();

        // I drop the inference_lock explicitly here because I think the rust
        // compiler might otherwise optimize and drop it early
        drop(inference_lock);
        godot_print!("[LLM Perf] Released inference lock");
    }

    godot_print!("[LLM Perf] Completion worker finished, channel closed");
    // We can't really throw an error here, since the other end of our channels seem to have died
    // but it's not `unreachable!()`, since we do end up here once the channels die.
    Ok(()) // accept our fate
}

/// Checks if the current generation should stop based on stop tokens.
/// This prevents the model from continuing after a stop sequence is detected.
///
/// # Arguments
/// * `last_n_tokens` - The last few tokens generated
/// * `stop_tokens` - List of token sequences that should stop generation
///
/// # Returns
/// * `should_stop` - Whether generation should stop
fn has_stop_tokens(last_n_tokens: &VecDeque<String>, stop_tokens: &[String]) -> bool {
    if last_n_tokens.is_empty() || stop_tokens.is_empty() {
        return false;
    }
    let last_n_concatenated: String = last_n_tokens.iter().fold(String::new(), |acc, x| acc + x);
    stop_tokens
        .iter()
        .any(|stop_token| last_n_concatenated.contains(stop_token))
}

pub enum EmbeddingsOutput {
    Embedding(Vec<f32>),
    FatalError(WorkerError),
}

pub fn run_embedding_worker(
    model: Arc<LlamaModel>,
    text_rx: Receiver<String>,
    embedding_tx: Sender<EmbeddingsOutput>,
) {
    if let Err(msg) = run_embedding_worker_result(model, text_rx, &embedding_tx) {
        embedding_tx
            .send(EmbeddingsOutput::FatalError(msg))
            .expect("Could not send llm worker fatal error back to consumer.");
    }
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
        let caps = detect_metal_capabilities();
        
        // Adjust thread count based on device capability
        let optimal_threads = match caps.gpu_type {
            GPUType::MetalHighPerformance => n_threads,
            GPUType::MetalIntegrated => n_threads,
            GPUType::MetalLowPower => std::cmp::min(n_threads, 4), // Limit threads on low-power devices
            _ => n_threads,
        };
        
        godot_print!("[LLM Embed] Using {} threads for embeddings", optimal_threads);
        
        let params = LlamaContextParams::default()
            .with_n_threads(optimal_threads)
            .with_embeddings(true);
            
        // Track the actual GPU layers being used
        metrics.update_gpu_layers(match caps.gpu_type {
            GPUType::MetalHighPerformance => u32::MAX,
            GPUType::MetalIntegrated => u32::MAX,
            GPUType::MetalLowPower => 20, // conservative estimate 
            _ => 0,
        });
        
        params
    };
    
    #[cfg(not(target_os = "ios"))]
    let ctx_params = {
        godot_print!("[LLM Embed] Using {} threads for embeddings on non-iOS", n_threads);
        metrics.update_gpu_layers(0); // Assume no GPU for non-iOS in tracking
        LlamaContextParams::default()
            .with_n_threads(n_threads)
            .with_embeddings(true)
    };

    godot_print!("[LLM Embed] Creating embedding context...");
    let context_start = Instant::now();
    let mut ctx = model.new_context(&LLAMA_BACKEND, ctx_params)?;
    let context_time = context_start.elapsed();
    godot_print!("[LLM Embed] Context created in {:.2}s", context_time.as_secs_f32());

    let mut embedding_count = 0;
    let mut total_tokens = 0;
    
    while let Ok(text) = text_rx.recv() {
        godot_print!("[LLM Embed] Processing text for embedding (length: {})", text.len());
        let embed_start = Instant::now();
        
        // HACK see comment in completion worker
        let inference_lock = GLOBAL_INFERENCE_LOCK
            .lock()
            .map_err(|_| WorkerError::GILPoisonError)?;
        godot_print!("[LLM Embed] Acquired inference lock");

        let mut batch = LlamaBatch::new(ctx.n_ctx() as usize, 1);
        
        let tokenize_start = Instant::now();
        let tokens = ctx.model.str_to_token(&text, AddBos::Always)?;
        let tokenize_time = tokenize_start.elapsed();
        
        godot_print!("[LLM Embed] Tokenized to {} tokens in {:.2}ms", 
            tokens.len(), tokenize_time.as_millis());
        total_tokens += tokens.len();
        
        metrics.update_context_size(tokens.len());
        metrics.update_batch_size(tokens.len());

        add_sequence(&mut batch, &tokens, 0, &[0]).expect("Failed to add sequence");

        ctx.clear_kv_cache();

        let decode_start = Instant::now();
        ctx.decode(&mut batch)?;
        let decode_time = decode_start.elapsed();
        metrics.record_decode_time(decode_time);
        
        godot_print!("[LLM Embed] Decoded batch in {:.2}ms", decode_time.as_millis());

        let embed_compute_start = Instant::now();
        let embedding = ctx.embeddings_seq_ith(0).unwrap().to_vec();
        let embed_compute_time = embed_compute_start.elapsed();
        
        godot_print!(
            "[LLM Embed] Computed embedding vector (size: {}) in {:.2}ms", 
            embedding.len(), 
            embed_compute_time.as_millis()
        );
        
        embedding_tx
            .send(EmbeddingsOutput::Embedding(embedding))
            .map_err(|_| WorkerError::SendError)?;
            
        embedding_count += 1;
        let total_time = embed_start.elapsed();
        
        godot_print!(
            "[LLM Embed] Total embedding time: {:.2}ms | Tokens/sec: {:.2}", 
            total_time.as_millis(),
            tokens.len() as f32 / (total_time.as_secs_f32().max(0.001))
        );

        drop(inference_lock);
        godot_print!("[LLM Embed] Released inference lock");
        
        // Log metrics periodically
        if embedding_count % 5 == 0 {
            metrics.log_basic_metrics();
            
            if get_log_level() >= LogLevel::Detailed {
                godot_print!(
                    "[LLM Embed] Processed {} embeddings with {} total tokens", 
                    embedding_count, 
                    total_tokens
                );
            }
        }
    }
    
    godot_print!(
        "[LLM Embed] Embedding worker finished | Total embeddings: {} | Total tokens: {}", 
        embedding_count, 
        total_tokens
    );
    metrics.log_all_metrics();
    
    Ok(())
}

fn dotproduct(a: &[f32], b: &[f32]) -> f32 {
    assert!(a.len() == b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let norm_a = dotproduct(a, a).sqrt();
    let norm_b = dotproduct(b, b).sqrt();
    if norm_a == 0. || norm_b == 0. {
        return f32::NAN;
    }
    dotproduct(a, b) / (norm_a * norm_b)
}

pub struct ModelLoader {
    source_path: String,
    dest_path: String,
    dump_progress: Arc<AtomicU32>,
    dump_completed: Arc<AtomicBool>,
    dump_thread: Option<std::thread::JoinHandle<()>>,
}

impl ModelLoader {
    pub fn new(source_path: String, dest_path: String) -> Self {
        Self {
            source_path,
            dest_path,
            dump_progress: Arc::new(AtomicU32::new(0)),
            dump_completed: Arc::new(AtomicBool::new(false)),
            dump_thread: None,
        }
    }

    pub fn start_loading(&mut self) {
        let source_path = self.source_path.clone();
        let dest_path = self.dest_path.clone();
        let progress = self.dump_progress.clone();
        let completed = self.dump_completed.clone();
        
        self.dump_thread = Some(std::thread::spawn(move || {
            // Check if source file exists
            if !FileAccess::file_exists(&source_path) {
                godot_error!("Source file does not exist: {}", source_path);
                return;
            }

            // Open source file using Godot's FileAccess
            let file_source = FileAccess::open(&source_path, ModeFlags::READ);
            let file_source = match file_source {
                Some(f) => f,
                None => {
                    godot_error!("Failed to open source file for reading: {}", source_path);
                    return;
                }
            };

            // Open destination file using Godot's FileAccess
            let file_dest = FileAccess::open(&dest_path, ModeFlags::WRITE);
            let mut file_dest = match file_dest {
                Some(f) => f,
                None => {
                    godot_error!("Failed to open destination file for writing: {}", dest_path);
                    return;
                }
            };

            // Get file length using Godot's API
            let length = file_source.get_length() as u64;
            let mut total_bytes_written = 0;
            let chunk_size: i64 = 8192;
            
            godot_print!("Starting to dump file of size: {} bytes", length);
            
            while total_bytes_written < length {
                let bytes_left = length - total_bytes_written;
                let to_read = std::cmp::min(bytes_left as i64, chunk_size);
                
                // Read chunk using Godot's API
                let buffer = file_source.get_buffer(to_read);
                if buffer.is_empty() {
                    godot_error!("Failed to read chunk at position: {}", total_bytes_written);
                    break;
                }

                // Write chunk using Godot's API
                file_dest.store_buffer(&buffer);
                total_bytes_written += buffer.len() as u64;
                
                // Store progress as integer percentage (0-100)
                let progress_pct = ((total_bytes_written as f64 / length as f64) * 100.0) as u32;
                progress.store(progress_pct, Ordering::Relaxed);
            }

            if total_bytes_written < length {
                godot_error!("Warning: Only dumped {} of {} bytes", total_bytes_written, length);
                return;
            }
            
            godot_print!("Successfully dumped all {} bytes", length);
            completed.store(true, Ordering::Relaxed);
        }));
    }

    pub fn is_completed(&self) -> bool {
        self.dump_completed.load(Ordering::Relaxed)
    }

    pub fn get_progress(&self) -> f32 {
        self.dump_progress.load(Ordering::Relaxed) as f32 / 100.0
    }

    pub fn wait_for_completion(&mut self) {
        if let Some(thread) = self.dump_thread.take() {
            thread.join().unwrap();
        }
    }
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

#[cfg(any(target_os = "ios", target_os = "macos"))]
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    device_model: String,
    system_memory_mb: u32,
    gpu_type: GPUType,
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            device_model: "Unknown".to_string(),
            system_memory_mb: 2048, // Conservative default
            gpu_type: GPUType::Unknown,
        }
    }
}

#[cfg(any(target_os = "ios", target_os = "macos"))]
pub fn get_device_info() -> DeviceInfo {
    use godot::classes::Os;
    
    let os = Os::singleton();
    let model_name = os.get_model_name().to_string();
    
    // Get memory info - this is an approximation as Godot doesn't expose exact memory
    let memory_mb = os.get_static_memory_usage() / (1024 * 1024);
    
    // Detect GPU capabilities
    let caps = detect_metal_capabilities();
    
    #[cfg(target_os = "ios")]
    godot_print!("[LLM iOS] Detected device: {}, Memory: {}MB", model_name, memory_mb);
    #[cfg(target_os = "macos")]
    godot_print!("[LLM macOS] Detected device: {}, Memory: {}MB", model_name, memory_mb);
    
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
        m if m.contains("iPad8,") || m.contains("iPad11,") => 24, // iPad Pro 2020/Air
        
        // Default values based on GPU type
        _ => match device_info.gpu_type {
            GPUType::MetalHighPerformance => 24,
            GPUType::MetalIntegrated => 16,
            GPUType::MetalLowPower => 8,
            _ => 0, // CPU only
        }
    };
    
    godot_print!("[LLM iOS] Optimized GPU layers for {}: {}", device_model, optimal_layers);
    optimal_layers
}

const MAX_TOKEN_STR_LEN: usize = 128;

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test_model_path {
        () => {
            std::env::var("TEST_MODEL")
                .unwrap_or("model.gguf".to_string())
                .as_str()
        };
    }

    macro_rules! test_embeddings_model_path {
        () => {
            std::env::var("TEST_EMBEDDINGS_MODEL")
                .unwrap_or("embeddings.gguf".to_string())
                .as_str()
        };
    }

    #[test]
    fn test_chat_completion() {
        let model = get_model(test_model_path!(), true).unwrap();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();

        let system_prompt = "You are a helpful assistant. The user asks you a question, and you provide an answer. You take multiple turns to provide the answer. Be consice and only provide the answer".to_string();
        std::thread::spawn(|| {
            run_completion_worker(
                model,
                prompt_rx,
                completion_tx,
                SamplerConfig::default(),
                4096,
                system_prompt,
                vec![],
                std::sync::mpsc::channel().0,
            )
        });

        prompt_tx
            .send("What is the capital of Denmark?".to_string())
            .unwrap();

        let result: String;
        loop {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(_)) => {}
                Ok(LLMOutput::Done(response)) => {
                    result = response;
                    break;
                }
                _ => unreachable!(),
            }
        }
        assert!(
            result.contains("Copenhagen"),
            "Expected completion to contain 'Copenhagen', got: {result}"
        );

        prompt_tx
            .send("What language to they speak there?".to_string())
            .unwrap();
        let result: String;
        loop {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(_)) => {}
                Ok(LLMOutput::Done(response)) => {
                    result = response;
                    break;
                }
                _ => unreachable!(),
            }
        }

        assert!(
            result.contains("Danish"),
            "Expected completion to contain 'Danish', got: {result}"
        );
    }

    #[test]
    fn test_embeddings() {
        let model = get_model(test_embeddings_model_path!(), true).unwrap();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (embedding_tx, embedding_rx) = std::sync::mpsc::channel();

        std::thread::spawn(|| run_embedding_worker(model, prompt_rx, embedding_tx));

        prompt_tx
            .send("Copenhagen is the capital of Denmark.".to_string())
            .unwrap();
        let copenhagen_embedding = match embedding_rx.recv() {
            Ok(EmbeddingsOutput::Embedding(vec)) => vec,
            _ => panic!(),
        };

        prompt_tx
            .send("Berlin is the capital of Germany.".to_string())
            .unwrap();
        let berlin_embedding = match embedding_rx.recv() {
            Ok(EmbeddingsOutput::Embedding(vec)) => vec,
            _ => panic!(),
        };

        prompt_tx
            .send("Your mother was a hamster and your father smelt of elderberries!".to_string())
            .unwrap();
        let insult_embedding = match embedding_rx.recv() {
            Ok(EmbeddingsOutput::Embedding(vec)) => vec,
            _ => panic!(),
        };

        assert!(
            insult_embedding.len() == berlin_embedding.len()
                && berlin_embedding.len() == copenhagen_embedding.len()
                && copenhagen_embedding.len() == insult_embedding.len(),
            "not all embedding lengths were equal"
        );

        // cosine similarity should not care about order
        assert_eq!(
            cosine_similarity(&copenhagen_embedding, &berlin_embedding),
            cosine_similarity(&berlin_embedding, &copenhagen_embedding)
        );

        // any vector should have cosine similarity 1 to itself
        // (tolerate small float error)
        assert!(
            (cosine_similarity(&copenhagen_embedding, &copenhagen_embedding) - 1.0).abs() < 0.001,
        );

        // the insult should have a lower similarity than the two geography sentences
        assert!(
            cosine_similarity(&copenhagen_embedding, &insult_embedding)
                < cosine_similarity(&copenhagen_embedding, &berlin_embedding)
        );
    }

    #[test]
    fn test_multiple_contexts_single_model() {
        let model = get_model(test_model_path!(), true).unwrap();

        let trivia_bot_system_prompt = "You are a trivia bot. You are asked a question, and you provide an answer. Be concise and only provide the answer".to_string();
        let (denmark_prompt_tx, denmark_prompt_rx) = std::sync::mpsc::channel();
        let (denmark_completion_tx, denmark_completion_rx) = std::sync::mpsc::channel();

        let model_clone = model.clone();
        std::thread::spawn(|| {
            run_completion_worker(
                model_clone,
                denmark_prompt_rx,
                denmark_completion_tx,
                SamplerConfig::default(),
                4096,
                trivia_bot_system_prompt,
                vec![],
                std::sync::mpsc::channel().0,
            )
        });

        let trivia_bot_system_prompt = "You are a trivia bot. You are asked a question, and you provide an answer. Be concise and only provide the answer".to_string();
        let (germany_prompt_tx, germany_prompt_rx) = std::sync::mpsc::channel();
        let (germany_completion_tx, germany_completion_rx) = std::sync::mpsc::channel();

        std::thread::spawn(|| {
            run_completion_worker(
                model,
                germany_prompt_rx,
                germany_completion_tx,
                SamplerConfig::default(),
                4096,
                trivia_bot_system_prompt,
                vec![],
                std::sync::mpsc::channel().0,
            )
        });

        denmark_prompt_tx
            .send("What is the capital of Denmark?".to_string())
            .unwrap();

        germany_prompt_tx
            .send("What is the capital of Germany?".to_string())
            .unwrap();

        // read dog output
        let result: String;
        loop {
            match denmark_completion_rx.recv() {
                Ok(LLMOutput::Token(_)) => {}
                Ok(LLMOutput::Done(response)) => {
                    result = response;
                    break;
                }
                _ => unreachable!(),
            }
        }
        assert!(
            result.to_lowercase().contains("copenhagen"),
            "Expected completion to contain 'Copenhagen', got: {result}"
        );

        // read cat output
        let result: String;
        loop {
            match germany_completion_rx.recv() {
                Ok(LLMOutput::Token(_)) => {}
                Ok(LLMOutput::Done(response)) => {
                    result = response;
                    break;
                }
                _ => unreachable!(),
            }
        }
        assert!(
            result.to_lowercase().contains("berlin"),
            "Expected completion to contain 'Berlin', got: {result}"
        );
    }

    #[test]
    fn test_context_shifting() {
        let model = get_model(test_model_path!(), true).unwrap();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();

        let system_prompt = "You are a helpful assistant.".to_string();
        std::thread::spawn(|| {
            run_completion_worker(
                model,
                prompt_rx,
                completion_tx,
                SamplerConfig::default(),
                100, // very low context size. will be exceeded immediately
                system_prompt,
                vec![],
                std::sync::mpsc::channel().0,
            )
        });

        prompt_tx
            .send("Please count down from 10 to 0, like this: Current 10, target 0. Current 9, target 0...".to_string())
            .unwrap();

        let result: String;
        loop {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(t)) => {
                    println!("new token: {t}");
                }
                Ok(LLMOutput::Done(response)) => {
                    result = response;
                    break;
                }
                Ok(LLMOutput::FatalErr(e)) => {
                    println!("got fatal error: {e}");
                    panic!();
                }
                _ => unreachable!(),
            }
        }
        assert!(
            result.contains("Current 1, target 0"),
            "Expected completion to contain 'Current 0, target 0', got: {result}"
        );
    }

    #[test]
    fn test_stop_tokens() {
        let model = get_model(test_model_path!(), true).unwrap();

        let (prompt_tx, prompt_rx) = std::sync::mpsc::channel();
        let (completion_tx, completion_rx) = std::sync::mpsc::channel();

        let system_prompt = "You are a helpful assistant.".to_string();
        std::thread::spawn(|| {
            run_completion_worker(
                model,
                prompt_rx,
                completion_tx,
                SamplerConfig::default(),
                4096,
                system_prompt,
                vec!["horse".to_string()], // Stop at "horse"
                std::sync::mpsc::channel().0,
            )
        });

        prompt_tx
            .send(
                "List these animals in alphabetical order: cat, dog, giraffe, horse, lion, mouse. Keep them in lowercase."
                    .to_string(),
            )
            .unwrap();

        let result: String;
        loop {
            match completion_rx.recv() {
                Ok(LLMOutput::Token(t)) => {
                    println!("new token: {t}");
                }
                Ok(LLMOutput::Done(response)) => {
                    result = response;
                    break;
                }
                Ok(LLMOutput::FatalErr(e)) => {
                    println!("got fatal error: {e}");
                    panic!();
                }
                _ => unreachable!(),
            }
        }

        assert!(
            result.to_lowercase().contains("giraffe"),
            "Expected output to contain text before stop token. Got: {result}"
        );
        assert!(
            result.to_lowercase().contains("horse"),
            "Expected output to contain stop token. Got: {result}"
        );
        assert!(
            !result.to_lowercase().contains("lion"),
            "Expected output to stop at stop token, but continued. Got: {result}"
        );
        assert!(
            !result.to_lowercase().contains("mouse"),
            "Expected output to stop at stop token, but continued. Got: {result}"
        );
    }
}
