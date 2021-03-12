#pragma once

#include "Ray.h"

struct HitRecord {
    Point3 p;
    Vec3 normal;
    double t;
    bool front_face;

    __device__ inline void set_face_normal(const Ray& r, const Vec3& outward_normal) {
        front_face = dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal : -outward_normal;
    }
};

class Hitable {
public:
    __device__ virtual bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const = 0;
};