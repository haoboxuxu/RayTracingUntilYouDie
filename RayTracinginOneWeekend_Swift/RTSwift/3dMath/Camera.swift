//
//  Camera.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/7.
//

import Foundation

class Camera {
    private var origin: Point3
    private var lowerLeftCorner: Point3
    private var horizontal: Vector3
    private var vertical: Vector3
    private var u: Vector3
    private var v: Vector3
    private var w: Vector3
    private var lensRadius: Float
    
    init(lookfrom: Point3, lookat: Point3,
         vup: Vector3 ,vfov: Float,
         aspectRatio: Float, aperture: Float,
         focusDist: Float) {
        
        let theta: Float = degrees2radians(degrees: vfov)
        let h: Float = tan(theta/2)
        let viewportHeight: Float = 2.0 * h
        let viewportWidth: Float = aspectRatio * viewportHeight
        
        w = (lookfrom - lookat).toUnitVector()
        u = cross(vup, w).toUnitVector()
        v = cross(w, u)
        
        origin = lookfrom
        horizontal = focusDist * viewportWidth * u
        vertical = focusDist * viewportHeight * v
        lowerLeftCorner = origin - horizontal/2 - vertical/2 - focusDist * w

        lensRadius = aperture / 2
    }
    
    func getRay(u: Float, v: Float) -> Ray {
        return Ray(origin: origin,
                   direction: lowerLeftCorner + u * horizontal + v * vertical - origin)
    }
}
