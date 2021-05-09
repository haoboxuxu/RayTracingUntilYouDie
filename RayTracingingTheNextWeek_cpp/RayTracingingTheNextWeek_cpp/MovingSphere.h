//
//  moving_sphere.h
//  RayTracingingTheNextWeek_cpp
//
//  Created by 徐浩博 on 2021/5/5.
//

#ifndef moving_sphere_h
#define moving_sphere_h

#include "utilitys.h"
#include "hittable.h"

class MovingSphere: public hittable {
public:
    MovingSphere() {}
    MovingSphere(
                 point3 cen0,
                 point3 cen1,
                 double _time0,
                 double _time1,
                 double r,
                 shared_ptr<Material> m
                 ) : center0(cen0), center1(cen1), time0(_time0), time1(_time1), radius(r), mat_ptr(m) {};
    bool hit(const Ray &r, double t_min, double t_max, hit_record &rec) const override;
    bool boundingBox(double time0, double time1, AABB &output_box) const override;
    
    point3 center(double time) const;
    
public:
    point3 center0, center1;
    double time0, time1;
    double radius;
    shared_ptr<Material> mat_ptr;
};

point3 MovingSphere::center(double time) const {
    return center0 + ((time - time0) / (time1 - time0))*(center1 - center0);
}

bool MovingSphere::hit(const Ray &r, double t_min, double t_max, hit_record &rec) const {
    Vec3 oc = r.origin() - center(r.time());
    auto a = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c = oc.length_squared() - radius*radius;
    
    auto discriminant = half_b*half_b - a*c;
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
    auto outward_normal = (rec.p - center(r.time())) / radius;
    rec.set_face_normal(r, outward_normal);
    rec.mat_ptr = mat_ptr;
    
    return true;
}

bool MovingSphere::boundingBox(double _time0, double _time1, AABB& output_box) const {
    AABB box0(
        center(_time0) - Vec3(radius, radius, radius),
        center(_time0) + Vec3(radius, radius, radius));
    AABB box1(
        center(_time1) - Vec3(radius, radius, radius),
        center(_time1) + Vec3(radius, radius, radius));
    output_box = surrounding_box(box0, box1);
    return true;
}


#endif /* moving_sphere_h */
