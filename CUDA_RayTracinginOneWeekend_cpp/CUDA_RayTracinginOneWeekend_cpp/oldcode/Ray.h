//
//  Ray.hpp
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/2/4.
//

#ifndef Ray_h
#define Ray_h

#include <stdio.h>
#include "Vec3.h"

class Ray {
public:
    __device__ Ray() {}
    __device__ Ray(const point3& origin, const Vec3& direction): orig(origin), dir(direction) {}

    __device__ point3 origin() const  { return orig; }
    __device__ Vec3 direction() const { return dir; }

    __device__ point3 at(double t) const { return orig + t*dir; }

public:
    point3 orig;
    Vec3 dir;
};


#endif /* Ray_h */
