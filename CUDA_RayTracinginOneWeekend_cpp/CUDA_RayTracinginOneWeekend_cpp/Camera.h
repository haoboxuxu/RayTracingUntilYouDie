//
//  Camera.h
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/3/8.
//

#ifndef Camera_h
#define Camera_h

#include <curand_kernel.h>
#include "utilitys.h"

class Camera {
public:
    __device__ Camera(
           point3 lookfrom,
           point3 lookat,
           Vec3   vup,
           double vfov, // vertical field-of-view in degrees
           double aspect_ratio,
           double aperture,
           double focus_dist
           ) {
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta/2);
        auto viewport_height = 2.0 * h;
        auto viewport_width = aspect_ratio * viewport_height;
        
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        
        origin = lookfrom;
        horizontal = focus_dist * viewport_width * u;
        vertical = focus_dist * viewport_height * v;
        lower_left_corner = origin - horizontal/2 - vertical/2 - focus_dist*w;
        
        lens_radius = aperture / 2;
    }
    
    
    __device__ Ray get_ray(double s, double t) const {
        Vec3 rd = lens_radius * random_in_unit_disk();
        Vec3 offset = u * rd.x() + v * rd.y();
        
        return Ray(origin + offset, lower_left_corner + s*horizontal + t*vertical - origin - offset);
    }
    
private:
    point3 origin;
    point3 lower_left_corner;
    Vec3 horizontal;
    Vec3 vertical;
    Vec3 u, v, w;
    double lens_radius;
};

#endif /* Camera_h */
