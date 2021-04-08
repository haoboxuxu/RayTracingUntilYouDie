//
//  MainView.swift
//  particles
//
//  Created by 徐浩博 on 2021/3/10.
//

import MetalKit
import simd

struct Particle {
    var color: float4
    var position: float2
    var verlocity: float2
}

class MainView: MTKView {
    
    var commandQueue: MTLCommandQueue!
    var clearPass: MTLComputePipelineState!
    var drawDotPass: MTLComputePipelineState!
    
    var particleBuffer: MTLBuffer!
    
    var screenSize: Float {
        return Float(self.bounds.width * 2)
    }
    
    var particleCount: Int = 1000
    
    required init(coder: NSCoder) {
        super.init(coder: coder)
        
        self.framebufferOnly = false
        
        self.device = MTLCreateSystemDefaultDevice()
        
        self.commandQueue = device?.makeCommandQueue()
        
        let library = device?.makeDefaultLibrary()
        let clearFunc = library?.makeFunction(name: "clear_pass_func")
        let drawDotFunc = library?.makeFunction(name: "draw_dots_func")
        
        do {
            clearPass = try device?.makeComputePipelineState(function: clearFunc!)
            drawDotPass = try device?.makeComputePipelineState(function: drawDotFunc!)
        } catch let err as NSError {
            print(err)
        }
        
        createParticles()
    }
    
    func createParticles() {
        var particles: [Particle] = []
        for _ in 0..<particleCount {
            let red: Float = Float(arc4random_uniform(100)) / 100
            let green: Float = Float(arc4random_uniform(100)) / 100
            let blue: Float = Float(arc4random_uniform(100)) / 100
            let particle = Particle(color: float4(red, green, blue, 1),
                                    position: float2(Float(arc4random_uniform(UInt32(screenSize))),
                                                     Float(arc4random_uniform(UInt32(screenSize)))),
                                    verlocity: float2((Float(arc4random() % 10) - 5) / 10, (Float(arc4random() % 10) - 5) / 10))
            particles.append(particle)
            print(particle.position.x, particle.position.y)
        }
        
        particleBuffer = device?.makeBuffer(bytes: particles, length: MemoryLayout<Particle>.stride * particleCount, options: [])
    }
    
    
    
    
    override func draw(_ dirtyRect: NSRect) {
        
        guard let drawable = self.currentDrawable else { return }
        
        let commandBuffer = commandQueue.makeCommandBuffer()
        let computeCommandEncoder = commandBuffer?.makeComputeCommandEncoder()
        
        computeCommandEncoder?.setComputePipelineState(clearPass)
        computeCommandEncoder?.setTexture(drawable.texture, index: 0)
        
        let w = clearPass.threadExecutionWidth                //32
        let h = clearPass.maxTotalThreadsPerThreadgroup / w   //8
        
        var threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
        var threadPerGrid = MTLSize(width: drawable.texture.width, height: drawable.texture.height, depth: 1)
        computeCommandEncoder?.dispatchThreads(threadPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeCommandEncoder?.setComputePipelineState(drawDotPass)
        computeCommandEncoder?.setBuffer(particleBuffer, offset: 0, index: 0)
        threadPerGrid = MTLSize(width: particleCount, height: 1, depth: 1)
        threadsPerThreadgroup = MTLSize(width: w, height: 1, depth: 1)
        computeCommandEncoder?.dispatchThreads(threadPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeCommandEncoder?.endEncoding()
        commandBuffer?.present(drawable)
        commandBuffer?.commit()
    }
}
