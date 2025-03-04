// Example of an optimized Metal kernel for iOS LLM inference
// This is a simplified version of a matrix multiplication kernel optimized for iOS devices

#include <metal_stdlib>
#include <metal_compute>
using namespace metal;

// Define smaller workgroup size for iOS devices
#define WORKGROUP_SIZE_X 8
#define WORKGROUP_SIZE_Y 8

// Optimized matrix multiplication kernel for iOS devices
kernel void mul_mat_f32_f32_optimized(
    device const float * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device       float * dst  [[buffer(2)]],
    constant  int64_t & ne00 [[buffer(3)]],
    constant  int64_t & ne01 [[buffer(4)]],
    constant  int64_t & ne02 [[buffer(5)]],
    constant  int64_t & ne10 [[buffer(6)]],
    constant  int64_t & ne11 [[buffer(7)]],
    constant  int64_t & ne12 [[buffer(8)]],
    constant  int64_t & ne0  [[buffer(9)]],
    constant  int64_t & ne1  [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3   ntg [[threads_per_threadgroup]])
{
    // Calculate indices
    const uint i0 = tgpig.x * ntg.x + tpitg.x;
    const uint i1 = tgpig.y * ntg.y + tpitg.y;
    
    // Check bounds
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    // Use smaller tile sizes for mobile GPUs
    const int tile_size = WORKGROUP_SIZE_X;
    
    // Use threadgroup memory for better performance
    threadgroup float tile_a[WORKGROUP_SIZE_Y][WORKGROUP_SIZE_X];
    threadgroup float tile_b[WORKGROUP_SIZE_Y][WORKGROUP_SIZE_X];
    
    float sum = 0.0f;
    
    // Process in tiles for better cache utilization
    for (uint t = 0; t < ne00; t += tile_size) {
        // Collaborative loading of tiles
        if (tpitg.x < tile_size && tpitg.y < tile_size) {
            const uint ta = min(t + tpitg.y, ne00 - 1);
            tile_a[tpitg.y][tpitg.x] = src0[i0*ne00 + ta];
            
            const uint tb = min(t + tpitg.x, ne00 - 1);
            tile_b[tpitg.y][tpitg.x] = src1[tb*ne1 + i1];
        }
        
        // Ensure all threads have loaded their data
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < tile_size && t + k < ne00; k++) {
            sum += tile_a[tpitg.y][k] * tile_b[k][tpitg.x];
        }
        
        // Ensure computation is complete before loading new tiles
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    dst[i0*ne1 + i1] = sum;
}

// Half-precision version for devices that support it
kernel void mul_mat_f16_f16_optimized(
    device const half * src0 [[buffer(0)]],
    device const half * src1 [[buffer(1)]],
    device       half * dst  [[buffer(2)]],
    constant  int64_t & ne00 [[buffer(3)]],
    constant  int64_t & ne01 [[buffer(4)]],
    constant  int64_t & ne02 [[buffer(5)]],
    constant  int64_t & ne10 [[buffer(6)]],
    constant  int64_t & ne11 [[buffer(7)]],
    constant  int64_t & ne12 [[buffer(8)]],
    constant  int64_t & ne0  [[buffer(9)]],
    constant  int64_t & ne1  [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3   ntg [[threads_per_threadgroup]])
{
    // Calculate indices
    const uint i0 = tgpig.x * ntg.x + tpitg.x;
    const uint i1 = tgpig.y * ntg.y + tpitg.y;
    
    // Check bounds
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    // Use smaller tile sizes for mobile GPUs
    const int tile_size = WORKGROUP_SIZE_X;
    
    // Use threadgroup memory for better performance
    threadgroup half tile_a[WORKGROUP_SIZE_Y][WORKGROUP_SIZE_X];
    threadgroup half tile_b[WORKGROUP_SIZE_Y][WORKGROUP_SIZE_X];
    
    half sum = 0.0h;
    
    // Process in tiles for better cache utilization
    for (uint t = 0; t < ne00; t += tile_size) {
        // Collaborative loading of tiles
        if (tpitg.x < tile_size && tpitg.y < tile_size) {
            const uint ta = min(t + tpitg.y, ne00 - 1);
            tile_a[tpitg.y][tpitg.x] = src0[i0*ne00 + ta];
            
            const uint tb = min(t + tpitg.x, ne00 - 1);
            tile_b[tpitg.y][tpitg.x] = src1[tb*ne1 + i1];
        }
        
        // Ensure all threads have loaded their data
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (uint k = 0; k < tile_size && t + k < ne00; k++) {
            sum += tile_a[tpitg.y][k] * tile_b[k][tpitg.x];
        }
        
        // Ensure computation is complete before loading new tiles
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write result
    dst[i0*ne1 + i1] = sum;
}

// SIMD-optimized version for high-end devices
kernel void mul_mat_f32_f32_simd_optimized(
    device const float * src0 [[buffer(0)]],
    device const float * src1 [[buffer(1)]],
    device       float * dst  [[buffer(2)]],
    constant  int64_t & ne00 [[buffer(3)]],
    constant  int64_t & ne01 [[buffer(4)]],
    constant  int64_t & ne02 [[buffer(5)]],
    constant  int64_t & ne10 [[buffer(6)]],
    constant  int64_t & ne11 [[buffer(7)]],
    constant  int64_t & ne12 [[buffer(8)]],
    constant  int64_t & ne0  [[buffer(9)]],
    constant  int64_t & ne1  [[buffer(10)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3   ntg [[threads_per_threadgroup]])
{
    // Calculate indices
    const uint i0 = tgpig.x * ntg.x + tpitg.x;
    const uint i1 = tgpig.y * ntg.y + tpitg.y;
    
    // Check bounds
    if (i0 >= ne0 || i1 >= ne1) {
        return;
    }
    
    // Use SIMD group for better performance on high-end devices
    simdgroup_float8x8 acc;
    
    // Initialize accumulator to zero
    for (uint i = 0; i < 8; ++i) {
        for (uint j = 0; j < 8; ++j) {
            acc[i][j] = 0.0f;
        }
    }
    
    // Process in chunks of 8x8
    for (uint k = 0; k < ne00; k += 8) {
        // Load 8x8 blocks from src0 and src1
        simdgroup_float8x8 a, b;
        
        // Load data into SIMD group
        for (uint i = 0; i < 8; ++i) {
            for (uint j = 0; j < 8; ++j) {
                uint kj = k + j;
                if (kj < ne00) {
                    a[i][j] = src0[(i0 + i) * ne00 + kj];
                    b[i][j] = src1[kj * ne1 + (i1 + j)];
                } else {
                    a[i][j] = 0.0f;
                    b[i][j] = 0.0f;
                }
            }
        }
        
        // Perform matrix multiplication using SIMD
        simdgroup_multiply(acc, a, b, acc);
    }
    
    // Write results
    for (uint i = 0; i < 8; ++i) {
        for (uint j = 0; j < 8; ++j) {
            if (i0 + i < ne0 && i1 + j < ne1) {
                dst[(i0 + i) * ne1 + (i1 + j)] = acc[i][j];
            }
        }
    }
} 