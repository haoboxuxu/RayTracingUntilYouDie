//
//  Lambertian.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/8.
//

import Foundation

class Lambertian: Material {
    
    var albedo: Color
    
    init(albedo: Color) {
        self.albedo = albedo
    }
    
    func scatter(ray: Ray, record: HitRecord, attenuation: inout Vector3, scattered: inout Ray) -> Bool {
        var scatterDirection = record.normal + Vector3.randomInUnitSphere()
        
        // Catch degenerate scatter direction
        if scatterDirection.isNearZero() {
            scatterDirection = record.normal
        }
        
        scattered = Ray(origin: record.point, direction: scatterDirection)
        attenuation = albedo
        return true
    }
}
