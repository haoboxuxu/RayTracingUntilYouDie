//
//  Vec3.hpp
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/2/4.
//

#ifndef Vec3_h
#define Vec3_h

#include <stdio.h>
#include <iostream>

class Vec3 {
public:
    Vec3() : e{0,0,0} {}
    Vec3(double e0, double e1, double e2) : e{e0, e1, e2} {}
    
    double x() const { return e[0]; }
    double y() const { return e[1]; }
    double z() const { return e[2]; }
    
    Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    double operator[](int i) const { return e[i]; }
    double& operator[](int i) { return e[i]; }
    
    Vec3& operator+=(const Vec3 &v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }
    
    Vec3& operator*=(const double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }
    
    Vec3& operator/=(const double t) {
        return *this *= 1/t;
    }
    
    double length() const {
        return sqrt(length_squared());
    }
    
    double length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }
    
    void write_color(std::ostream &out) {
        // Write the translated [0,255] value of each color component.
        out << static_cast<int>(255.999 * e[0]) << ' '
        << static_cast<int>(255.999 * e[1]) << ' '
        << static_cast<int>(255.999 * e[2]) << '\n';
    }
    
public:
    double e[3];
};

#endif /* Vec3_h */
