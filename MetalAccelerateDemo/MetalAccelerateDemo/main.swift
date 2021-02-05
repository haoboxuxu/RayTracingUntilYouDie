//
//  main.swift
//  MetalAccelerateDemo
//
//  Created by 徐浩博 on 2021/2/6.
//

import MetalKit


let count: Int = 1000000

var array1 = genRandomArray()
var array2 = genRandomArray()

cpuWay(arr1: array1, arr2: array2)
gpuWay(arr1: array1, arr2: array2)

func cpuWay(arr1: [Float], arr2: [Float]) {
    print("CPU Way")
    let startTime = CFAbsoluteTimeGetCurrent()
    var res = [Float].init(repeating: 0.0, count: count)
    
    for i in 0..<count {
        res[i] = arr1[i] + arr2[i]
    }
    let endTime = CFAbsoluteTimeGetCurrent()
    
    print("time = \(String(format: "%.05f", endTime - startTime)) seconds")
}

func gpuWay(arr1: [Float], arr2: [Float]) {
    print("GPU Way")
    let startTime = CFAbsoluteTimeGetCurrent()
    
    let device = MTLCreateSystemDefaultDevice()
    let commandQueue = device?.makeCommandQueue()
    let gpuFunctionLibrary = device?.makeDefaultLibrary()
    let addTwoArraysFunction = gpuFunctionLibrary?.makeFunction(name: "addTwoArrays")
    var computePipelineState: MTLComputePipelineState!
    do {
        computePipelineState = try device?.makeComputePipelineState(function: addTwoArraysFunction!)
    } catch {
        print(error)
    }
    
    let arr1Buffer = device?.makeBuffer(bytes: arr1,
                                        length: MemoryLayout<Float>.size * count,
                                        options: .storageModeShared)
    
    let arr2Buffer = device?.makeBuffer(bytes: arr2,
                                        length: MemoryLayout<Float>.size * count,
                                        options: .storageModeShared)
    
    let resBuffer = device?.makeBuffer(length: MemoryLayout<Float>.size * count,
                                       options: .storageModeShared)
    
    let commandBuffer = commandQueue?.makeCommandBuffer()
    let commandEncoder = commandBuffer?.makeComputeCommandEncoder()
    commandEncoder?.setComputePipelineState(computePipelineState)
    
    commandEncoder?.setBuffer(arr1Buffer, offset: 0, index: 0)
    commandEncoder?.setBuffer(arr2Buffer, offset: 0, index: 1)
    commandEncoder?.setBuffer(resBuffer, offset: 0, index: 2)
    
    let threadsPerGrid = MTLSize(width: count, height: 1, depth: 1)
    let maxThreadsPerThreadgroup = computePipelineState.maxTotalThreadsPerThreadgroup // 1024
    let threadsPerThreadgroup = MTLSize(width: maxThreadsPerThreadgroup, height: 1, depth: 1)
    commandEncoder?.dispatchThreads(threadsPerGrid,
                                    threadsPerThreadgroup: threadsPerThreadgroup)
    
    commandEncoder?.endEncoding()
    commandBuffer?.commit()
    commandBuffer?.waitUntilCompleted()
    var resultBufferPointer = resBuffer?.contents().bindMemory(to: Float.self,
                                                               capacity: MemoryLayout<Float>.size * count)
    
    let endTime = CFAbsoluteTimeGetCurrent()
    
    print("time = \(String(format: "%.05f", endTime - startTime)) seconds")
    
    for i in 0..<3 {
        print("\(arr1[i]) + \(arr2[i]) = \(Float(resultBufferPointer!.pointee) as Any)")
        resultBufferPointer = resultBufferPointer?.advanced(by: 1)
    }
}

func genRandomArray() -> [Float] {
    var res = [Float].init(repeating: 0.0, count: count)
    for i in 0..<count {
        res[i] = Float(arc4random_uniform(10))
    }
    return res
}
