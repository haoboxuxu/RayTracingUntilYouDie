#pragma once

#include "Hitable.h"

class HitableList : public Hitable {
public:
	__device__ HitableList() {}
	__device__ HitableList(Hitable** object, int len) { objects = object; objects_len = len; }
	__device__ virtual bool hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const;
public:
    Hitable** objects;
    int objects_len;
};

__device__ bool HitableList::hit(const Ray& r, float tmin, float tmax, HitRecord& rec) const {
    HitRecord temp_rec;
    bool hit_anything = false;
    float closest_so_far = tmax;

    for (int i = 0; i < objects_len; i++) {
        if (objects[i]->hit(r, tmin, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}