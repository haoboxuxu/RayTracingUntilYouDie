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
#include "Vec3.cpp"

class Ray {
public:
    Vec3 orig;
    Vec3 dir;

    Ray() {}
    Ray(const Vec3& origin, const Vec3 direction): orig(origin), dir(direction) {}
    Vec3 origin() const { return origin(); }
    Vec3 direction() const { return dir; }
    Vec3 at(double t) const {
        return orig + t * dir;
    }
};


#endif /* Ray_h */
