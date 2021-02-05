//
//  hittable.h
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/2/4.
//

#ifndef hittable_h
#define hittable_h

#include "utilitys.h"
#include "Ray.h"

class Material;

struct hit_record {
    point3 p;
    Vec3 normal;
    shared_ptr<Material> mat_ptr;
    double t;
    bool front_face;
    inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class hittable {
public:
    virtual bool hit(const Ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

#endif /* hittable_h */
