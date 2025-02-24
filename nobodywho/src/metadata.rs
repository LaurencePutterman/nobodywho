use serde::{Deserialize, Serialize};
use std::path::Path;
use godot::classes::FileAccess;
use godot::classes::file_access::ModeFlags;

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
        if !FileAccess::file_exists(path) {
            return None;
        }

        let file = FileAccess::open(path, ModeFlags::READ)?;
        let size = file.get_length() as u64;
        let modified = FileAccess::get_modified_time(path) as u64;
        let filename = Path::new(path).file_name()?.to_string_lossy().to_string();

        Some(FileMetadata {
            size,
            modified,
            filename,
        })
    }
}

pub fn load_metadata(metadata_path: &str) -> StoredMetadata {
    if !FileAccess::file_exists(metadata_path) {
        return StoredMetadata::default();
    }

    let file = match FileAccess::open(metadata_path, ModeFlags::READ) {
        Some(f) => f,
        None => return StoredMetadata::default(),
    };

    match serde_json::from_str(&file.get_as_text().to_string()) {
        Ok(metadata) => metadata,
        Err(_) => StoredMetadata::default(),
    }
}

pub fn save_metadata(metadata_path: &str, metadata: &StoredMetadata) {
    if let Ok(json) = serde_json::to_string_pretty(metadata) {
        if let Some(mut file) = FileAccess::open(metadata_path, ModeFlags::WRITE) {
            file.store_string(json.as_str());
        }
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
    let should_dump = !FileAccess::file_exists(dest_path) || 
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