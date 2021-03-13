#pragma once

#include "Hitable.h"

class Sphere: public Hitable {
public:
	__device__ Sphere() {}
	__device__ Sphere(Point3 cen, float r, Material* m) : center(cen), radius(r), mat_ptr(m) {};
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
public:
	Point3 center;
	float radius;
    Material* mat_ptr;
};

__device__ bool Sphere::hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const {

    Vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < tmin || tmax < root) {
        root = (-half_b + sqrtd) / a;
        if (root < tmin || tmax < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    Vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;

    return true;
}