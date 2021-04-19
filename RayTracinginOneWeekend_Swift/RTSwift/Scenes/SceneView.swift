//
//  SceneView.swift
//  RTSwift
//
//  Created by 徐浩博 on 2021/2/9.
//

import Foundation

class SceneView {
    private var world: HitableList
    
    init() {
        world = HitableList()
        setUpScene()
    }
    
    func getScene() -> HitableList {
        return self.world
    }
    
    func setUpScene() {
        let ground_material = Lambertian(albedo: Color(0.5, 0.5, 0.5))
        world.objects.append(Sphere(center: Point3(0,-1000,0), radius: 1000, material: ground_material))
        
        for i in -11..<11 {
            for j in -11..<11 {
                
                let a = Float(i)
                let b = Float(j)
                
                let choose_mat = randomFloat()
                let center = Point3(a + 0.9*randomFloat(), 0.2, b + 0.9*randomFloat())
                
                if ((center - Point3(4, 0.2, 0)).length > 0.9) {
                    var sphere_material: Material
                    
                    if (choose_mat < 0.8) {
                        // diffuse
                        let albedo = Color.random() * Color.random()
                        sphere_material = Lambertian(albedo: albedo)
                        world.objects.append(Sphere(center: center, radius: 0.2, material: sphere_material))
                    } else if (choose_mat < 0.95) {
                        // metal
                        let albedo = Color.random(0.5, 1)
                        let fuzz = randomFloat(0, 0.5)
                        sphere_material = Metal(albedo: albedo, fuzz: fuzz)
                        world.objects.append(Sphere(center: center, radius: 0.2, material: sphere_material))
                    } else {
                        // glass
                        sphere_material = Dielectric(indexOfRefraction: 1.5)
                        world.objects.append(Sphere(center: center, radius: 0.22, material: sphere_material))
                    }
                }
            }
        }
        
        let material1 = Dielectric(indexOfRefraction: 1.5)
        world.objects.append(Sphere(center: Point3(0, 1, 0), radius: 1.0, material: material1))
        
        let material2 = Lambertian(albedo: Color(0.4, 0.2, 0.1))
        world.objects.append(Sphere(center: Point3(-4, 1, 0), radius: 1.0, material: material2))
        
        let material3 = Metal(albedo: Color(0.7, 0.6, 0.5), fuzz: 0.0)
        world.objects.append(Sphere(center: Point3(4, 1, 0), radius: 1.0, material: material3))
        
    }
}
