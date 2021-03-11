#pragma once

#include "Hitable.h"

class Sphere: public Hitable {
public:
	__device__ Sphere() {}
	__device__ Sphere(Point3 cen, double r) : center(cen), radius(r) {};
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
public:
	Point3 center;
	float radius;
};

__device__ bool Sphere::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    Vec3 oc = r.origin() - center;
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0) return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    rec.normal = (rec.p - center) / radius;

    return true;
}