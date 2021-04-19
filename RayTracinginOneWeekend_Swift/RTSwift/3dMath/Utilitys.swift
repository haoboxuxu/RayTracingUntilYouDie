//
//  Utilitys.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/6.
//

import Foundation
import simd

let infinity: Float = Float.infinity
let pi: Float = 3.1415926535897932385

func randomFloat() -> Float {
    return Float(drand48())
}

func randomFloat(_ min: Float, _ max: Float) -> Float {
    return Float.random(in: min...max)
}

func clamp(_ x: Float, _ min: Float, _ max: Float) -> Float {
    if x < min {
        return min
    }
    if x > max {
        return max
    }
    return x
}

func degrees2radians(degrees: Float) -> Float {
    return degrees * pi / 180.0
}

// MARK: extension for vector3
func dot(_ left: Vector3, _ right: Vector3) -> Float {
    return simd_dot(left.elm, right.elm)
}

func cross(_ left: Vector3, _ right: Vector3) -> Vector3 {
    return Vector3(simd_cross(left.elm, right.elm))
}
