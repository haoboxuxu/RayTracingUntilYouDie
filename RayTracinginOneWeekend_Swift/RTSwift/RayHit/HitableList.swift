//
//  HitableList.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/7.
//

import Foundation

class HitableList: Hitable {
    var objects: [Hitable] = []
    
    func hit(ray: Ray, tMin: Float, tMax: Float, record: inout HitRecord) -> Bool {
        var tempRecord = HitRecord()
        var hitAnything: Bool = false
        var closestSoFar = tMax
        
        for object in objects {
            if object.hit(ray: ray, tMin: tMin, tMax: closestSoFar, record: &tempRecord) {
                hitAnything = true
                closestSoFar = tempRecord.t
                record = tempRecord
            }
        }
        
        return hitAnything;
    }
    
    
}
