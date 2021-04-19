//
//  Hitable.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/7.
//

import Foundation

class HitRecord {
    var point: Point3
    var normal: Vector3
    var material: Material?
    var t: Float
    var frontFace: Bool
    
    func setFaceNormal(ray: Ray, outwardNormal: Vector3) {
        frontFace = dot(ray.direction, outwardNormal) < 0;
        normal = frontFace ? outwardNormal : -outwardNormal
    }
    
    init() {
        t = 0
        point = Point3(0, 0, 0)
        normal = Vector3(0, 0, 0)
        frontFace = true
    }
}

protocol Hitable {
    func hit(ray: Ray, tMin: Float, tMax: Float, record: inout HitRecord) -> Bool
}
