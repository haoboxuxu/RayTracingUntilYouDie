//
//  Vec3.hpp
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/3/8.
//

#ifndef Vec3_h
#define Vec3_h
#include <cmath>
#include <iostream>
using std::sqrt;

class Vec3 {
public:
    float e[3];

public:
    __host__ __device__ Vec3() : e{0,0,0} {}
    __host__ __device__ Vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}
    
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    //__host__ __device__ inline const Vec3& operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }

    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; }
    
    __host__ __device__ inline Vec3& operator+=(const Vec3& v);
    __host__ __device__ inline Vec3& operator-=(const Vec3& v);
    __host__ __device__ inline Vec3& operator*=(const Vec3& v);
    __host__ __device__ inline Vec3& operator/=(const Vec3& v);
    __host__ __device__ inline Vec3& operator*=(const float t);
    __host__ __device__ inline Vec3& operator/=(const float t);
    
    
    
    __host__ __device__ inline float length() const { return sqrt(length_squared()); }
    
    __host__ __device__ inline float length_squared() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];}
    
    inline static Vec3 random() {
        return Vec3(random_float(), random_float(), random_float());
    }
    
    inline static Vec3 random(float min, float max) {
        return Vec3(random_float(min,max), random_float(min,max), random_float(min,max));
    }
    
    bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        const auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }
};

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const Vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const Vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) {
    return *this *= 1 / t;
}

// Type aliases for Vec3
using point3 = Vec3;   // 3D point
using color = Vec3;    // RGB color

// Vec3 Utility Functions
inline std::ostream& operator<<(std::ostream &out, const Vec3 &v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline Vec3 operator+(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline Vec3 operator-(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline Vec3 operator*(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline Vec3 operator*(float t, const Vec3 &v) {
    return Vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
}

inline Vec3 operator*(const Vec3 &v, float t) {
    return t * v;
}

inline Vec3 operator/(Vec3 v, float t) {
    return (1/t) * v;
}

inline float dot(const Vec3 &u, const Vec3 &v) {
    return u.e[0] * v.e[0]
    + u.e[1] * v.e[1]
    + u.e[2] * v.e[2];
}

inline Vec3 cross(const Vec3 &u, const Vec3 &v) {
    return Vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}




Vec3 random_in_unit_sphere() {
    while (true) {
        auto p = Vec3::random(-1,1);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

Vec3 random_unit_vector() {
    return unit_vector(random_in_unit_sphere());
}

Vec3 random_in_hemisphere(const Vec3& normal) {
    Vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2*dot(v,n)*n;
}

Vec3 refract(const Vec3& uv, const Vec3& n, float etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    Vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    Vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ Vec3 random_in_unit_disk() {
    while (true) {
        auto p = Vec3(random_float(-1,1), random_float(-1,1), 0);
        if (p.length_squared() >= 1) continue;
        return p;
    }
}

#endif /* Vec3_h */
