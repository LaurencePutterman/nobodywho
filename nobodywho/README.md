# NobodyWho

A Godot extension for AI functionality using llama.cpp.

## Building

### iOS Build Configuration

When building for iOS, you can customize the bundle identifier in two ways:

1. Using an environment variable:
```bash
BUNDLE_ID="com.yourcompany.nobodywho" ./build_ios.sh
```

2. Using a command-line argument:
```bash
./build_ios.sh --bundle-id "com.yourcompany.nobodywho"
```

If no bundle identifier is specified, it will default to `org.godot.nobodywho`.

To see all available options:
```bash
./build_ios.sh --help
```

### Building for All Platforms

To build for all supported platforms:

```bash
./build.sh
```

To include Android builds:
```bash
./build.sh --android
```

## Requirements

- Rust and Cargo (install from https://rustup.rs)
- Godot 4.3 or newer
- For iOS builds:
  - macOS with Xcode and Xcode Command Line Tools
  - iOS SDK
- For Android builds:
  - Android NDK
  - `cargo-ndk` (install with `cargo install cargo-ndk`)
  - Set `ANDROID_NDK_HOME` environment variable

## License

[Add your license information here] 