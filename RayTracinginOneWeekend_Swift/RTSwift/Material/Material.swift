//
//  Material.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/8.
//

import Foundation

protocol Material {
    func scatter(ray: Ray, record: HitRecord, attenuation: inout Vector3, scattered: inout Ray) -> Bool
}
