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
    Ray() {}
    Ray(const point3& origin, const Vec3& direction, double time = 0.0)
        : orig(origin), dir(direction), tm(time)
    {}
    
    point3 origin() const  { return orig; }
    Vec3 direction() const { return dir; }
    double time() const { return tm; }
    
    point3 at(double t) const {
        return orig + t*dir;
    }
public:
    point3 orig;
    Vec3 dir;
    double tm;
};


#endif /* Ray_h */
