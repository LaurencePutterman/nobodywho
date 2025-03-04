use godot::prelude::*;
use crate::sampler_config::{SamplerConfig, SamplerMethod, Temperature, TopK, TopP, MinP, TypicalP, MirostatV2};

/// Returns a sampler configuration optimized for speed on iOS devices
pub fn get_speed_optimized_sampler_config() -> SamplerConfig {
    SamplerConfig {
        method: SamplerMethod::Temperature(Temperature {
            seed: 1234,
            temperature: 0.7,
        }),
        penalty_last_n: 64,
        penalty_repeat: 1.1,
        penalty_freq: 0.0,
        penalty_present: 0.0,
        use_grammar: false,
        gbnf_grammar: String::new(),
    }
}

/// Returns a sampler configuration optimized for balanced performance on iOS devices
pub fn get_balanced_sampler_config() -> SamplerConfig {
    SamplerConfig {
        method: SamplerMethod::TopP(TopP {
            seed: 1234,
            top_p: 0.9,
            min_keep: 0,
        }),
        penalty_last_n: 64,
        penalty_repeat: 1.1,
        penalty_freq: 0.0,
        penalty_present: 0.0,
        use_grammar: false,
        gbnf_grammar: String::new(),
    }
}

/// Returns a sampler configuration optimized for quality on iOS devices
pub fn get_quality_optimized_sampler_config() -> SamplerConfig {
    SamplerConfig {
        method: SamplerMethod::MirostatV2(MirostatV2 {
            seed: 1234,
            temperature: 0.8,
            tau: 5.0,
            eta: 0.1,
        }),
        penalty_last_n: 64,
        penalty_repeat: 1.1,
        penalty_freq: 0.0,
        penalty_present: 0.0,
        use_grammar: false,
        gbnf_grammar: String::new(),
    }
}

/// Returns the appropriate sampler configuration based on the performance profile
pub fn get_optimized_sampler_config(performance_profile: &str) -> SamplerConfig {
    match performance_profile {
        "speed" => get_speed_optimized_sampler_config(),
        "quality" => get_quality_optimized_sampler_config(),
        _ => get_balanced_sampler_config(),
    }
} 