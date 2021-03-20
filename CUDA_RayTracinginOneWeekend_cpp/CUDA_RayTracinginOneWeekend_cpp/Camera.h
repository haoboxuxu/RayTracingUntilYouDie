#pragma once
//
//  Camera.h
//  CUDA_RayTracinginOneWeekend_cpp
//
//  Created by ÐìºÆ²© on 2021/3/12.
//

#include "Ray.h"

class Camera {
public:
    __device__ Camera(Point3 lookfrom, Point3 lookat, Vec3 vup, float vfov, float aspect_ratio) {
        float theta = vfov * 3.1415926535897932385 / 180;  //degrees_to_radians(vfov);
        float h = tan(theta / 2);
        float viewport_height = 2.0 * h;
        float viewport_width = aspect_ratio * viewport_height;

        float focal_length = 1.0;

        Vec3 w = unit_vector(lookfrom - lookat);
        Vec3 u = unit_vector(cross(vup, w));
        Vec3 v = cross(w, u);

        origin = lookfrom;
        horizontal = viewport_width * u;
        vertical = viewport_height * v;
        lower_left_corner = origin - horizontal / 2 - vertical / 2 - w;
    }

    __device__ Ray get_ray(float s, float t) const {
        return Ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
    }
private:
    Point3 origin;
    Point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
};