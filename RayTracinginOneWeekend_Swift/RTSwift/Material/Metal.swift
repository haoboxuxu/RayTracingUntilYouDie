//
//  Metal.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/8.
//

import Foundation

class Metal: Material {
    var albedo: Color
    var fuzz: Float
    
    init(albedo: Color, fuzz: Float) {
        self.albedo = albedo
        self.fuzz = min(max(0.0, fuzz), 1.0)
    }
    
    func scatter(ray: Ray, record: HitRecord, attenuation: inout Vector3, scattered: inout Ray) -> Bool {
        let reflect = Vector3.reflect(v: ray.direction, n: record.normal).toUnitVector()
        scattered = Ray(origin: record.point, direction: reflect + fuzz * Vector3.randomInUnitSphere())
        attenuation = albedo
        return (dot(scattered.direction, record.normal) > 0);
    }
    
    
}
