//
//  add2arrs.metal
//  MetalAccelerateDemo
//
//  Created by 徐浩博 on 2021/2/6.
//

#include <metal_stdlib>
using namespace metal;

kernel void addTwoArrays(constant float *arr1  [[ buffer(0)]],
                         constant float *arr2  [[ buffer(1)]],
                         device   float *res   [[ buffer(2) ]],
                                  uint   index [[ thread_position_in_grid ]]) {
    res[index] = arr1[index] + arr2[index];
}
