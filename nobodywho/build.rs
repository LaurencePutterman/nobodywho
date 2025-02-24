use std::env;

fn main() {
    let target = env::var("TARGET").unwrap();
    let mut config = cmake::Config::new("llama.cpp");
    
    config
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("GGML_BLAS", "OFF");

    if target.contains("ios") {
        let sdk = if target.contains("x86_64-apple-ios") {
            "iphonesimulator"
        } else {
            "iphoneos"
        };

        let sdk_path = std::process::Command::new("xcrun")
            .args(&["--sdk", sdk, "--show-sdk-path"])
            .output()
            .expect("failed to execute xcrun")
            .stdout;
        let sdk_path = String::from_utf8_lossy(&sdk_path).trim().to_string();

        config
            .define("CMAKE_SYSTEM_NAME", "iOS")
            .define("CMAKE_OSX_SYSROOT", &sdk_path)
            .define("CMAKE_OSX_DEPLOYMENT_TARGET", "13.0")
            .define("CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH", "NO")
            .define("CMAKE_IOS_INSTALL_COMBINED", "YES")
            .define("GGML_OPENMP", "OFF") // Disable OpenMP for iOS
            .define("CMAKE_SYSTEM_VERSION", "13.0")
            .define("CMAKE_OSX_ARCHITECTURES", if target.contains("x86_64") { "x86_64" } else { "arm64" })
            .define("CMAKE_C_FLAGS", format!("-isysroot {} -mios-version-min=13.0", sdk_path))
            .define("CMAKE_CXX_FLAGS", format!("-isysroot {} -mios-version-min=13.0", sdk_path))
            .define("CMAKE_BUILD_TYPE", "Release")
            .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
            .generator("Unix Makefiles");

        println!("cargo:rustc-env=IPHONEOS_DEPLOYMENT_TARGET=13.0");
        println!("cargo:rustc-link-search=native={}/usr/lib", sdk_path);
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Foundation");
    } else {
        config.define("GGML_OPENMP", "ON");
    }

    config.build();
} 