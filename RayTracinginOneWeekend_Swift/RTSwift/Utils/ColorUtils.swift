//
//  Color.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/7.
//

import Foundation

class Color8Bit {
    var e: [UInt8]
    
    init() {
        e = [0, 0, 0]
    }
    
    convenience init(_ r: UInt8, _ g: UInt8, _ b: UInt8) {
        self.init()
        self.e[0] = r
        self.e[1] = g
        self.e[2] = b
    }
    
    var r: UInt8 { e[0] }
    var g: UInt8 { e[1] }
    var b: UInt8 { e[2] }
}

class ColorUtils {
    
    static func to8BitColor(pixelColor: Color, samplesPerPixel: Int) -> Color8Bit {
        var r = pixelColor.x
        var g = pixelColor.y
        var b = pixelColor.z
        
        let scale: Float = 1.0 / Float(samplesPerPixel)
        r = sqrt(scale * r)
        g = sqrt(scale * g)
        b = sqrt(scale * b)
        
        return Color8Bit(UInt8(255.999 * r),
                         UInt8(255.999 * g),
                         UInt8(255.999 * b))
    }
    
    static func hitSphere(center: Point3, radius: Float, ray: Ray) -> Float {
        let originCenter = ray.origin - center
        let a = dot(ray.direction, ray.direction)
        let half_b = dot(originCenter, ray.direction)
        let c = originCenter.lengthSquared - radius * radius
        let discriminant = half_b * half_b - a * c
        if discriminant < 0 {
            return -1.0
        } else {
            return (-half_b - sqrt(discriminant) ) / (2.0*a)
        }
    }
    
    static func rayColor(ray: Ray, world: Hitable, depth: Int) -> Color {
        var record = HitRecord()
        
        if depth <= 0 {
            return Color(0,0,0)
        }
        
        if world.hit(ray: ray, tMin: 0.001, tMax: infinity, record: &record) {
            
            var scattered = Ray()
            var attenuation = Color()
            
            if ((record.material?.scatter(ray: ray,
                                          record: record,
                                          attenuation: &attenuation,
                                          scattered: &scattered)) != nil) {
                return attenuation * rayColor(ray: scattered, world: world, depth: depth-1)
            }
            return Color(0,0,0);
        }
        let unitDirection = ray.direction.toUnitVector()
        let t = 0.5 * (unitDirection.y + 1.0)
        return (1.0 - t) * Color(1.0, 1.0, 1.0) + t * Color(0.5, 0.7, 1.0)
    }
}
