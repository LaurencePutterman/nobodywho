#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Optimization settings
#ifndef WORKGROUP_SIZE
#define WORKGROUP_SIZE 128
#endif

#ifndef TILE_SIZE
#define TILE_SIZE 8
#endif

// Matrix multiplication kernel for float precision
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
    const int64_t i02 = tpig.z;
    const int64_t i01 = tpig.y;
    const int64_t i11 = tpig.x;

    const int64_t i = i02*ne01*ne11 + i01*ne11 + i11;

    const int64_t i10 = i / ne11;
    const int64_t i00 = i / ne10;

    if (i00 >= ne00 || i10 >= ne10) {
        return;
    }

    // Compute matrix multiplication for this block
    float sum = 0.0f;
    for (int k = 0; k < ne00; k++) {
        sum += src0[i00*ne00 + k] * src1[k*ne10 + i10];
    }
    dst[i] = sum;
}

// Matrix multiplication kernel for half precision
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
    const int64_t i02 = tpig.z;
    const int64_t i01 = tpig.y;
    const int64_t i11 = tpig.x;

    const int64_t i = i02*ne01*ne11 + i01*ne11 + i11;

    const int64_t i10 = i / ne11;
    const int64_t i00 = i / ne10;

    if (i00 >= ne00 || i10 >= ne10) {
        return;
    }

    // Compute matrix multiplication for this block
    half sum = 0.0h;
    for (int k = 0; k < ne00; k++) {
        sum += src0[i00*ne00 + k] * src1[k*ne10 + i10];
    }
    dst[i] = sum;
}

// Tiled matrix multiplication for better performance
kernel void matmul_f32_tiled(
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
    
    // Check bounds
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    // Compute tile-based matrix multiplication
    for (int i = 0; i < min(TILE_SIZE, ne0 - i0); i++) {
        for (int j = 0; j < min(TILE_SIZE, ne1 - i1); j++) {
            float sum = 0.0f;
            for (int k = 0; k < ne00; k++) {
                sum += src0[(i0 + i) * ne00 + k] * src1[k * ne1 + (i1 + j)];
            }
            dst[(i0 + i) * ne1 + (i1 + j)] = sum;
        }
    }
}

// SIMD-optimized matrix multiplication for high-end devices
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
    const int i0 = tpig.y * 8; // Using 8x8 tiles for SIMD
    const int i1 = tpig.x * 8;
    
    // Check bounds
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    // Use thread-local storage for accumulation
    float acc[8][8] = {0.0f};
    
    // Compute tile-based matrix multiplication with manual unrolling
    for (int k = 0; k < ne00; k++) {
        for (int i = 0; i < min(8, ne0 - i0); i++) {
            float a_val = src0[(i0 + i) * ne00 + k];
            for (int j = 0; j < min(8, ne1 - i1); j++) {
                float b_val = src1[k * ne1 + (i1 + j)];
                acc[i][j] += a_val * b_val;
            }
        }
    }
    
    // Write results back to global memory
    for (int i = 0; i < min(8, ne0 - i0); i++) {
        for (int j = 0; j < min(8, ne1 - i1); j++) {
            dst[(i0 + i) * ne1 + (i1 + j)] = acc[i][j];
        }
    }
}
