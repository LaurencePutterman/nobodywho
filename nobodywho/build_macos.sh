#!/bin/bash

# Exit on error
set -e

# Enable debug output
set -x

# Setup variables
FRAMEWORK_NAME="nobodywho"
MACOS_SDK_VERSION=$(xcrun --sdk macosx --show-sdk-version)
MACOS_SDK_PATH=$(xcrun --sdk macosx --show-sdk-path)
MIN_MACOS_VERSION="10.15"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf target/universal-macos
rm -rf target/*/debug/build
rm -rf target/*/release/build

# Create directories
mkdir -p target/universal-macos/frameworks

# Create framework directory structure for debug and release
mkdir -p target/universal-macos/frameworks/nobodywho-debug.framework/Headers
mkdir -p target/universal-macos/frameworks/nobodywho-release.framework/Headers

# Create debug and release Info.plist files
cat > target/universal-macos/frameworks/nobodywho-debug.framework/Info.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>nobodywho-debug</string>
    <key>CFBundleIdentifier</key>
    <string>com.maxlaurence.nobodywho</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>nobodywho-debug</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>10.15</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
</dict>
</plist>
EOF

cat > target/universal-macos/frameworks/nobodywho-release.framework/Info.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>nobodywho-release</string>
    <key>CFBundleIdentifier</key>
    <string>com.maxlaurence.nobodywho</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>nobodywho-release</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>10.15</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>MacOSX</string>
    </array>
</dict>
</plist>
EOF

# Create exports file
echo "_gdext_rust_init" > target/exports.txt

# Build for macOS (debug)
echo "Building for macOS (debug)..."
cargo build

# Build for macOS (release)
echo "Building for macOS (release)..."
cargo build --release

# Function to convert static lib to dynamic and fix symbols
process_library() {
    local input="$1"
    local output="$2"
    local is_debug="$3"
    local sdk_path=$(xcrun -sdk macosx --show-sdk-path)
    local toolchain_path=$(xcrun -find clang | xargs dirname)/../
    
    echo "Processing library: $input -> $output"
    
    # Check if input exists
    if [ ! -f "$input" ]; then
        echo "Error: Input file $input does not exist!"
        exit 1
    fi
    
    # Determine install name based on debug/release
    local install_name
    if [ "$is_debug" = true ]; then
        install_name="@rpath/nobodywho-debug.framework/nobodywho-debug"
    else
        install_name="@rpath/nobodywho-release.framework/nobodywho-release"
    fi
    
    # Create a dynamic library
    xcrun -sdk macosx clang++ -dynamiclib \
        -install_name "$install_name" \
        -current_version 1.0 \
        -compatibility_version 1.0 \
        -exported_symbols_list target/exports.txt \
        -o "$output" \
        "$input" \
        -framework Foundation \
        -framework Metal \
        -framework MetalKit \
        -framework Accelerate \
        -lc++ \
        -isysroot "$sdk_path" \
        -mmacosx-version-min=$MIN_MACOS_VERSION \
        -g \
        -Wl,-no_deduplicate \
        -Wl,-platform_version,macos,$MIN_MACOS_VERSION,$MACOS_SDK_VERSION

    # Check if output was created
    if [ ! -f "$output" ]; then
        echo "Error: Failed to create output file $output"
        exit 1
    fi

    # Strip debug symbols to separate file
    xcrun strip -S "$output" -o "${output}.stripped"
    mv "${output}.stripped" "$output"

    # Generate debug symbols with proper structure
    dsymutil -o "${output}.dSYM" "$output"
    
    # Verify dSYM was created
    if [ ! -d "${output}.dSYM" ]; then
        echo "Error: Failed to create dSYM bundle"
        exit 1
    fi
}

# Process libraries
echo "Processing libraries..."
process_library "target/debug/libnobodywho.a" \
    "target/debug/libnobodywho-debug.dylib" \
    true

process_library "target/release/libnobodywho.a" \
    "target/release/libnobodywho-release.dylib" \
    false

# Copy files to final location with proper names
echo "Copying files to final location..."
cp "target/debug/libnobodywho-debug.dylib" "target/universal-macos/frameworks/nobodywho-debug.framework/nobodywho-debug"
cp "target/release/libnobodywho-release.dylib" "target/universal-macos/frameworks/nobodywho-release.framework/nobodywho-release"

# Copy debug symbols
cp -R "target/debug/libnobodywho-debug.dylib.dSYM" "target/universal-macos/frameworks/nobodywho-debug.framework/nobodywho-debug.dSYM"
cp -R "target/release/libnobodywho-release.dylib.dSYM" "target/universal-macos/frameworks/nobodywho-release.framework/nobodywho-release.dSYM"

# Set permissions
chmod 755 target/universal-macos/frameworks/nobodywho-debug.framework/nobodywho-debug
chmod 755 target/universal-macos/frameworks/nobodywho-release.framework/nobodywho-release

echo "macOS build completed successfully!"