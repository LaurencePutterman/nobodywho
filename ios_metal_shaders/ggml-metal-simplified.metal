#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>

using namespace metal;

// Include iOS optimizations if available
#if __has_include("ios_optimizations.h")
#include "ios_optimizations.h"
#else
// Default values
#define WORKGROUP_SIZE 128
#define TILE_SIZE 8
#define USE_SIMD 0
#define USE_FP16 1
#endif

// Optimized matrix multiplication kernel for float
kernel void matmul_f32(
    device const float * src0,
    device const float * src1,
    device float * dst,
    constant int & ne00,
    constant int & ne01,
    constant int & ne02,
    constant int & ne10,
    constant int & ne11,
    constant int & ne12,
    constant int & ne0,
    constant int & ne1,
    uint3 tpig[[thread_position_in_grid]],
    uint3 tptg[[threads_per_threadgroup]],
    uint3 tpitg[[thread_position_in_threadgroup]])
{
    const int i0 = tpig.y;
    const int i1 = tpig.x;
    
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    float sum = 0.0f;
    for (int k = 0; k < ne00; k++) {
        sum += src0[i0*ne00 + k] * src1[k*ne1 + i1];
    }
    dst[i0*ne1 + i1] = sum;
}

// Optimized matrix multiplication kernel for half
#if USE_FP16
kernel void matmul_f16(
    device const half * src0,
    device const half * src1,
    device half * dst,
    constant int & ne00,
    constant int & ne01,
    constant int & ne02,
    constant int & ne10,
    constant int & ne11,
    constant int & ne12,
    constant int & ne0,
    constant int & ne1,
    uint3 tpig[[thread_position_in_grid]],
    uint3 tptg[[threads_per_threadgroup]],
    uint3 tpitg[[thread_position_in_threadgroup]])
{
    const int i0 = tpig.y;
    const int i1 = tpig.x;
    
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    half sum = 0.0h;
    for (int k = 0; k < ne00; k++) {
        sum += src0[i0*ne00 + k] * src1[k*ne1 + i1];
    }
    dst[i0*ne1 + i1] = sum;
}
#endif

// SIMD-optimized version for high-end devices
#if USE_SIMD
kernel void matmul_f32_simd(
    device const float * src0,
    device const float * src1,
    device float * dst,
    constant int & ne00,
    constant int & ne01,
    constant int & ne02,
    constant int & ne10,
    constant int & ne11,
    constant int & ne12,
    constant int & ne0,
    constant int & ne1,
    uint3 tpig[[thread_position_in_grid]],
    uint3 tptg[[threads_per_threadgroup]],
    uint3 tpitg[[thread_position_in_threadgroup]])
{
    const int i0 = tpig.y * TILE_SIZE;
    const int i1 = tpig.x * TILE_SIZE;
    
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    float acc[TILE_SIZE][TILE_SIZE] = {0.0f};
    
    for (int k = 0; k < ne00; k++) {
        for (int i = 0; i < min(TILE_SIZE, ne0 - i0); i++) {
            float a_val = src0[(i0 + i) * ne00 + k];
            for (int j = 0; j < min(TILE_SIZE, ne1 - i1); j++) {
                float b_val = src1[k * ne1 + (i1 + j)];
                acc[i][j] += a_val * b_val;
            }
        }
    }
    
    for (int i = 0; i < min(TILE_SIZE, ne0 - i0); i++) {
        for (int j = 0; j < min(TILE_SIZE, ne1 - i1); j++) {
            dst[(i0 + i) * ne1 + (i1 + j)] = acc[i][j];
        }
    }
}
#endif
