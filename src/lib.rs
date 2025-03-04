mod chat_state;
mod db;
mod llm;
mod metadata;
mod sampler_config;
mod sampler_resource;
#[cfg(target_os = "ios")]
mod ios_optimizations;
#[cfg(target_os = "ios")]
mod metal_shaders;

use godot::classes::{INode, ProjectSettings, FileAccess};
use godot::prelude::*;
use godot::obj::Base;
use llm::{run_completion_worker, run_embedding_worker};
use sampler_resource::NobodyWhoSampler;
use std::sync::mpsc::{Receiver, Sender};

#[godot_api]
impl NobodyWhoChat {
    fn get_model(&mut self) -> Result<llm::Model, String> {
        let gd_model_node = self.model_node.as_mut().ok_or("Model node was not set")?;
        let mut nobody_model = gd_model_node.bind_mut();
        nobody_model.load_model().map_err(|e| e.to_string())
    }

    fn get_sampler_config(&mut self) -> sampler_config::SamplerConfig {
        #[cfg(target_os = "ios")]
        {
            // On iOS, check if we should use an optimized sampler configuration
            if let Some(gd_sampler) = self.sampler.as_mut() {
                // User has explicitly set a sampler, use that
                let nobody_sampler: GdRef<NobodyWhoSampler> = gd_sampler.bind();
                nobody_sampler.sampler_config.clone()
            } else {
                // No sampler set, use an optimized one based on device state
                use llm::ThermalController;
                let thermal_controller = ThermalController::new();
                let _ = thermal_controller.update();
                let profile = thermal_controller.get_performance_profile();
                
                godot_print!("[LLM Chat] Using iOS-optimized sampler for performance profile: {:?}", profile);
                ios_optimizations::get_sampler_for_performance_profile(&profile)
            }
        }
        
        #[cfg(not(target_os = "ios"))]
        {
            if let Some(gd_sampler) = self.sampler.as_mut() {
                let nobody_sampler: GdRef<NobodyWhoSampler> = gd_sampler.bind();
                nobody_sampler.sampler_config.clone()
            } else {
                // Default sampler configuration
                sampler_config::SamplerConfig::default()
            }
        }
    }

    #[func]
    fn _ready(&mut self) {
        #[cfg(target_os = "ios")]
        {
            // Initialize Metal optimizations for iOS
            metal_shaders::configure_metal_parameters();
            godot_print!("[LLM Chat] Initialized iOS Metal optimizations");
        }
    }

    // ... rest of the existing implementation ...
}

// ... rest of the existing code ... 