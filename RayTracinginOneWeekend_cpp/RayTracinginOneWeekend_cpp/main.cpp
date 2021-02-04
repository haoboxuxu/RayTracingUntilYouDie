//
//  main.cpp
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/2/4.
//

#include <iostream>
#include <fstream>
#include "utilitys.h"
#include "Color.h"
#include "hittable_list.h"
#include "Sphere.h"
using namespace std;

color ray_color(const Ray& r, const hittable& world) {
    hit_record rec;
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1,1,1));
    }
    Vec3 unit_direction = unit_vector(r.direction());
    auto t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}

int main() {
    ofstream fout("/Users/haoboxuxu/Desktop/output.ppm");
    if (fout.fail()) {
        cout << "open file output.ppm fail" << endl;
        return -1;
    }
    
    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 1440;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    
    // World
    hittable_list world;
    world.add(make_shared<Sphere>(point3(0,0,-1), 0.5));
    world.add(make_shared<Sphere>(point3(0,-100.5,-1), 100));
    
    // Camera
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;
    
    auto origin = point3(0, 0, 0);
    auto horizontal = Vec3(viewport_width, 0, 0);
    auto vertical = Vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal/2 - vertical/2 - Vec3(0, 0, focal_length);
    
    // Render
    fout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height-1; j >= 0; --j) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; ++i) {
            auto u = double(i) / (image_width-1);
            auto v = double(j) / (image_height-1);
            Ray r(origin, lower_left_corner + u*horizontal + v*vertical - origin);
            color pixel_color = ray_color(r, world);
            write_color(fout, pixel_color);
        }
    }
    
    std::cerr << "\nDone.\n";
    
    fout.close();
    
    return 0;
}
