//
//  Dielectric.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/9.
//

import Foundation

class Dielectric: Material {
    
    var ir: Float
    
    init(indexOfRefraction: Float) {
        self.ir = indexOfRefraction
    }
    
    func scatter(ray: Ray, record: HitRecord, attenuation: inout Vector3, scattered: inout Ray) -> Bool {
        attenuation = Color(1.0, 1.0, 1.0)
        let refractionRatio = record.frontFace ? (1.0 / ir) : ir
        
        let unitDirection = ray.direction.toUnitVector()
        
        let cos_theta = fmin(dot(-unitDirection, record.normal), 1.0)
        let sin_theta = sqrt(1.0 - cos_theta*cos_theta)
        
        let cannot_refract = refractionRatio * sin_theta > 1.0;
        let direction: Vector3
        
        if cannot_refract {
            direction = Vector3.reflect(v: unitDirection, n: record.normal)
        } else {
            direction = Vector3.refract(uv: unitDirection, n: record.normal, etai_over_etat: refractionRatio)
        }
        
        scattered = Ray(origin: record.point, direction: direction)
        return true
    }
    
    private static func reflectance(cos: Float, reflectIndex: Float) -> Float {
        var r0 = (1 - reflectIndex) / (1 + reflectIndex)
        r0 *= r0
        return r0 * (1 - r0) * pow(1 - cos, 5)
    }
}
