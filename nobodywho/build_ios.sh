#!/bin/bash

# Exit on error
set -e

# Enable debug output
set -x

# Setup variables
FRAMEWORK_NAME="nobodywho"
IOS_SDK_VERSION=$(xcrun --sdk iphoneos --show-sdk-version)
IOS_SDK_PATH=$(xcrun --sdk iphoneos --show-sdk-path)
MIN_IOS_VERSION="14.0"

# llama.cpp configuration
# To update the commit hash:
# 1. Visit https://github.com/ggerganov/llama.cpp/commits/master
# 2. Choose a stable commit (preferably one that's been tested)
# 3. Use either:
#    - Full hash (e.g., 7a2c913e66353362d7f28d612fd3c9d51a831eda)
#    - Short hash (e.g., 7a2c913)
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_REV="7a2c913e66353362d7f28d612fd3c9d51a831eda"  # Full hash for precise version control

# Default bundle identifier (can be overridden)
BUNDLE_ID="${BUNDLE_ID:-org.godot.nobodywho}"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to setup llama.cpp
setup_llama_cpp() {
    # Check if git is available
    if ! command_exists git; then
        echo "Error: git is required to setup llama.cpp"
        exit 1
    fi

    if [ ! -d "llama.cpp" ]; then
        echo "Cloning llama.cpp..."
        git clone $LLAMA_CPP_REPO llama.cpp
        (cd llama.cpp && git checkout $LLAMA_CPP_REV)
    else
        echo "llama.cpp directory already exists, checking version..."
        (cd llama.cpp && \
         current_rev=$(git rev-parse --short HEAD) && \
         if [ "$current_rev" != "$LLAMA_CPP_REV" ]; then
             echo "Updating llama.cpp to required version..." && \
             git fetch && \
             git checkout $LLAMA_CPP_REV
         fi)
    fi
}

# Help message
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --bundle-id <id>    Set the bundle identifier (default: $BUNDLE_ID)"
    echo "  --help             Show this help message"
    echo ""
    echo "You can also set the bundle identifier using the BUNDLE_ID environment variable."
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --bundle-id)
            BUNDLE_ID="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "Using bundle identifier: $BUNDLE_ID"

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf target/universal-ios
rm -rf target/aarch64-apple-ios/debug/build
rm -rf target/aarch64-apple-ios/release/build

# Setup llama.cpp dependency
echo "Setting up llama.cpp dependency..."
setup_llama_cpp

# Create directories
mkdir -p target/universal-ios/frameworks

# Create framework directory structure for debug and release
mkdir -p target/universal-ios/frameworks/nobodywho-debug.framework/Headers
mkdir -p target/universal-ios/frameworks/nobodywho-release.framework/Headers

# Function to create Info.plist
create_info_plist() {
    local output_file="$1"
    local executable_name="$2"
    
    cat > "$output_file" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleExecutable</key>
    <string>$executable_name</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>$executable_name</string>
    <key>CFBundlePackageType</key>
    <string>FMWK</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>MinimumOSVersion</key>
    <string>$MIN_IOS_VERSION</string>
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>iPhoneOS</string>
    </array>
    <key>UIDeviceFamily</key>
    <array>
        <integer>1</integer>
        <integer>2</integer>
    </array>
</dict>
</plist>
EOF
}

# Create Info.plist files
create_info_plist "target/universal-ios/frameworks/nobodywho-debug.framework/Info.plist" "nobodywho-debug"
create_info_plist "target/universal-ios/frameworks/nobodywho-release.framework/Info.plist" "nobodywho-release"

# Create exports file
echo "_gdext_rust_init" > target/exports.txt

# Build for iOS arm64
echo "Building for iOS arm64..."
SDKROOT=$IOS_SDK_PATH \
IPHONEOS_DEPLOYMENT_TARGET=$MIN_IOS_VERSION \
CFLAGS="-isysroot $IOS_SDK_PATH -mios-version-min=$MIN_IOS_VERSION -target arm64-apple-ios -g" \
CXXFLAGS="-isysroot $IOS_SDK_PATH -mios-version-min=$MIN_IOS_VERSION -target arm64-apple-ios -g" \
cargo build --target aarch64-apple-ios

# Build for iOS arm64 (release)
echo "Building for iOS arm64 (release)..."
SDKROOT=$IOS_SDK_PATH \
IPHONEOS_DEPLOYMENT_TARGET=$MIN_IOS_VERSION \
CFLAGS="-isysroot $IOS_SDK_PATH -mios-version-min=$MIN_IOS_VERSION -target arm64-apple-ios -g" \
CXXFLAGS="-isysroot $IOS_SDK_PATH -mios-version-min=$MIN_IOS_VERSION -target arm64-apple-ios -g" \
cargo build --target aarch64-apple-ios --release

# Function to convert static lib to dynamic and fix symbols
process_library() {
    local input="$1"
    local output="$2"
    local is_debug="$3"
    local sdk_path=$(xcrun -sdk iphoneos --show-sdk-path)
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
    
    # Create a dynamic library with proper install name and rpaths
    xcrun -sdk iphoneos clang++ -dynamiclib \
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
        -L"$toolchain_path/lib/clang/16/lib/darwin/ios" \
        -mios-version-min=$MIN_IOS_VERSION \
        -target arm64-apple-ios \
        -g \
        -Wl,-no_deduplicate \
        -Wl,-objc_abi_version,2 \
        -Wl,-platform_version,ios,$MIN_IOS_VERSION,$IOS_SDK_VERSION \
        -Wl,-rpath,@executable_path/Frameworks \
        -Wl,-rpath,@loader_path/Frameworks

    # Check if output was created
    if [ ! -f "$output" ]; then
        echo "Error: Failed to create output file $output"
        exit 1
    fi

    # Sign the library with ad-hoc signature
    codesign --force --sign - --timestamp=none "$output"

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
echo "Processing arm64 libraries..."
process_library "target/aarch64-apple-ios/debug/libnobodywho.a" \
    "target/aarch64-apple-ios/debug/libnobodywho-debug.dylib" \
    true

process_library "target/aarch64-apple-ios/release/libnobodywho.a" \
    "target/aarch64-apple-ios/release/libnobodywho-release.dylib" \
    false

# Copy files to final location with proper names
echo "Copying files to final location..."
cp "target/aarch64-apple-ios/debug/libnobodywho-debug.dylib" "target/universal-ios/frameworks/nobodywho-debug.framework/nobodywho-debug"
cp "target/aarch64-apple-ios/release/libnobodywho-release.dylib" "target/universal-ios/frameworks/nobodywho-release.framework/nobodywho-release"

# Copy debug symbols
cp -R "target/aarch64-apple-ios/debug/libnobodywho-debug.dylib.dSYM" "target/universal-ios/frameworks/nobodywho-debug.framework.dSYM"
cp -R "target/aarch64-apple-ios/release/libnobodywho-release.dylib.dSYM" "target/universal-ios/frameworks/nobodywho-release.framework.dSYM"

# Set permissions
chmod 755 target/universal-ios/frameworks/nobodywho-debug.framework/nobodywho-debug
chmod 755 target/universal-ios/frameworks/nobodywho-release.framework/nobodywho-release

# Sign the frameworks
codesign --force --sign - --timestamp=none target/universal-ios/frameworks/nobodywho-debug.framework
codesign --force --sign - --timestamp=none target/universal-ios/frameworks/nobodywho-release.framework

# Update gdextension file to point to the new framework locations
sed -i '' 's|res://addons/nobodywho/nobodywho-universal-apple-ios-debug.dylib|@rpath/nobodywho-debug.framework/nobodywho-debug|g' nobodywho.gdextension
sed -i '' 's|res://addons/nobodywho/nobodywho-universal-apple-ios-release.dylib|@rpath/nobodywho-release.framework/nobodywho-release|g' nobodywho.gdextension

echo "iOS build completed successfully!" 