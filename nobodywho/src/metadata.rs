use godot::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::SystemTime;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileMetadata {
    pub size: u64,
    pub modified: u64,
    pub filename: String,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct StoredMetadata {
    pub dumped_models: std::collections::HashMap<String, FileMetadata>,
}

impl FileMetadata {
    pub fn from_path(path: &str) -> Option<Self> {
        let path = Path::new(path);
        if !path.exists() {
            return None;
        }

        let metadata = fs::metadata(path).ok()?;
        let modified = metadata.modified().ok()?
            .duration_since(SystemTime::UNIX_EPOCH).ok()?
            .as_secs();

        Some(FileMetadata {
            size: metadata.len(),
            modified,
            filename: path.file_name()?.to_string_lossy().to_string(),
        })
    }
}

pub fn load_metadata(metadata_path: &str) -> StoredMetadata {
    if !Path::new(metadata_path).exists() {
        return StoredMetadata::default();
    }

    match fs::read_to_string(metadata_path) {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => StoredMetadata::default(),
    }
}

pub fn save_metadata(metadata_path: &str, metadata: &StoredMetadata) {
    if let Ok(json) = serde_json::to_string_pretty(metadata) {
        let _ = fs::write(metadata_path, json);
    }
}

pub fn should_dump_model(
    source_path: &str,
    dest_path: &str,
    metadata_path: &str,
    model_key: &str,
) -> bool {
    let mut stored_metadata = load_metadata(metadata_path);
    
    // Get current source metadata
    let source_metadata = match FileMetadata::from_path(source_path) {
        Some(metadata) => metadata,
        None => return true, // Should dump if we can't read source metadata
    };

    // Check if destination exists and metadata matches source
    let should_dump = !Path::new(dest_path).exists() || 
        stored_metadata.dumped_models.get(model_key)
            .map(|stored| {
                stored.size != source_metadata.size ||
                stored.modified != source_metadata.modified ||
                stored.filename != source_metadata.filename
            })
            .unwrap_or(true);

    if should_dump {
        // Save new metadata using source file's metadata
        stored_metadata.dumped_models.insert(model_key.to_string(), source_metadata);
        save_metadata(metadata_path, &stored_metadata);
    }

    should_dump
} 