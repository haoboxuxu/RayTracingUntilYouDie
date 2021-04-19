//
//  main.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/6.
//

import Foundation

let renderer = ImageRenderer(imageWidth: imageWidth,
                             imageHeight: imageHeight)

// Camera
let lookfrom = Point3(13,2,3)
let lookat = Point3(0,0,0)
let vup = Vector3(0,1,0)
let dist2focus: Float = 10.0
let aperture: Float = 0.1

let camera = Camera(lookfrom: lookfrom,
                    lookat: lookat,
                    vup: vup,
                    vfov: 20.0,
                    aspectRatio: aspectRatio,
                    aperture: aperture,
                    focusDist: dist2focus)

// World
let sceneView = SceneView()
let world = sceneView.getScene()


for j in (0...renderer.imageHeight-1).reversed() {
    
    print("Scanlines remaining: \(j)")
    
    for i in 0...renderer.imageWidth-1 {
        
        var pixelColor = Color(0, 0, 0)
        
        for _ in 0..<samplesPerPixel {
            let u = (Float(i) + randomFloat()) / Float(imageWidth-1)
            let v = (Float(j) + randomFloat()) / Float(imageHeight-1)
            let ray = camera.getRay(u: u, v: v)
            pixelColor += ColorUtils.rayColor(ray: ray, world: world, depth: maxDepth)
        }
        
        renderer.setColor(x: i, y: j, color8Bit: ColorUtils.to8BitColor(pixelColor: pixelColor, samplesPerPixel: samplesPerPixel))
    }
}

renderer.savePNGImage(imageName: "output_swift.png", rootURL: "/Users/haoboxuxu/Desktop")
