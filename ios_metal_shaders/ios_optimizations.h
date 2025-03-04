#ifndef IOS_OPTIMIZATIONS_H
#define IOS_OPTIMIZATIONS_H

// Optimization flags for different iOS devices
// HIGH_END: A14 and newer (iPhone 12+)
// MID_RANGE: A12-A13 (iPhone XS/XR to iPhone 11)
// LOW_END: A11 and older (iPhone X and older)

#if defined(HIGH_END)
    #define WORKGROUP_SIZE 256
    #define TILE_SIZE 8
    #define USE_SIMD 1
    #define USE_FP16 1
#elif defined(MID_RANGE)
    #define WORKGROUP_SIZE 128
    #define TILE_SIZE 8
    #define USE_SIMD 0
    #define USE_FP16 1
#elif defined(LOW_END)
    #define WORKGROUP_SIZE 64
    #define TILE_SIZE 4
    #define USE_SIMD 0
    #define USE_FP16 0
#else
    // Default optimizations
    #define WORKGROUP_SIZE 128
    #define TILE_SIZE 8
    #define USE_SIMD 0
    #define USE_FP16 1
#endif

#endif // IOS_OPTIMIZATIONS_H
