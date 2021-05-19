//
//  main.cpp
//  RayTracingingTheNextWeek_cpp
//
//  Created by 徐浩博 on 2021/2/4.
//

#include <iostream>
#include <fstream>
#include "utilitys.h"
#include "Color.h"
#include "hittable_list.h"
#include "Sphere.h"
#include "Material.h"
#include "Camera.h"
#include "Material.h"
#include "RenderConfig.h"
#include "MovingSphere.h"
#include "AABB.h"
#include "bvh.h"
#include "texture.h"
using namespace std;

hittable_list two_spheres() {
    hittable_list objects;
    
    auto checker = make_shared<CheckerTexture>(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
    
    objects.add(make_shared<Sphere>(point3(0,-10, 0), 10, make_shared<Lambertian>(checker)));
    objects.add(make_shared<Sphere>(point3(0, 10, 0), 10, make_shared<Lambertian>(checker)));
    
    return objects;
}

hittable_list random_scene() {
    hittable_list world;
    
    auto checker = make_shared<CheckerTexture>(color(0.2, 0.3, 0.1), color(0.9, 0.9, 0.9));
    world.add(make_shared<Sphere>(point3(0,-1000,0), 1000, make_shared<Lambertian>(checker)));
    
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());
            
            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                shared_ptr<Material> sphere_material;
                
                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = make_shared<Lambertian>(albedo);
                    auto center2 = center + Vec3(0, random_double(0,.5), 0);
                    world.add(make_shared<MovingSphere>(center, center2, 0.0, 1.0, 0.2, sphere_material));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = make_shared<Metal>(albedo, fuzz);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                } else {
                    // glass
                    sphere_material = make_shared<Dielectric>(1.5);
                    world.add(make_shared<Sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }
    
    auto material1 = make_shared<Dielectric>(1.5);
    world.add(make_shared<Sphere>(point3(0, 1, 0), 1.0, material1));
    
    auto material2 = make_shared<Lambertian>(color(0.4, 0.2, 0.1));
    world.add(make_shared<Sphere>(point3(-4, 1, 0), 1.0, material2));
    
    auto material3 = make_shared<Metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(make_shared<Sphere>(point3(4, 1, 0), 1.0, material3));
    
    return world;
}

// loop
color ray_color(const Ray& r, const hittable& world, int depth) {
    hit_record rec;
    
    // If we've exceeded the ray bounce limit, no more light is gathered.
    if (depth <= 0) {
        return color(0,0,0);
    }
    
    if (world.hit(r, 0.001, infinity, rec)) {
        Ray scattered;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            return attenuation * ray_color(scattered, world, depth-1);
        }
        return color(0,0,0);
    }
    
    Vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}



int main() {
    ofstream fout("/Users/haoboxuxu/Desktop/output.ppm");
    if (fout.fail()) {
        cout << "open file output.ppm failed" << endl;
        return -1;
    }
    
    // Image
    
    // World
    hittable_list world;
    
    point3 lookfrom;
    point3 lookat;
    auto vfov = 40.0;
    auto aperture = 0.0;
    
    switch (0) {
        case 1:
            world = random_scene();
            lookfrom = point3(13,2,3);
            lookat = point3(0,0,0);
            vfov = 20.0;
            aperture = 0.1;
            break;
        default:
        case 2:
            world = two_spheres();
            lookfrom = point3(13,2,3);
            lookat = point3(0,0,0);
            vfov = 20.0;
            break;
    }
    
    // Camera
    Vec3 vup(0,1,0);
    auto dist_to_focus = 10.0;

    Camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, 0.0, 1.0);
    
    // Render
    fout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            color pixel_color(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_double()) / (image_width-1);
                auto v = (j + random_double()) / (image_height-1);
                Ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(fout, pixel_color, samples_per_pixel);
        }
    }
    
    std::cerr << "\nDone.\n";
    
    fout.close();
    
    return 0;
}
