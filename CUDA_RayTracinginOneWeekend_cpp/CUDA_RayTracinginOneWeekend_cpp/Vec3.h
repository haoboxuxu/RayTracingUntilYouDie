#pragma once

#include "cuda_runtime.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>

class Vec3 {
public:
	__host__ __device__ Vec3() {}
	__host__ __device__ Vec3(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const Vec3& operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const { return Vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline Vec3& operator+=(const Vec3& v2);
    __host__ __device__ inline Vec3& operator-=(const Vec3& v2);
    __host__ __device__ inline Vec3& operator*=(const Vec3& v2);
    __host__ __device__ inline Vec3& operator/=(const Vec3& v2);
    __host__ __device__ inline Vec3& operator*=(const float t);
    __host__ __device__ inline Vec3& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __host__ __device__ inline float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline void make_unit_vector();
    __host__ __device__ inline bool near_zero() const;

public:
	float e[3];
};

using Point3 = Vec3;
using Color = Vec3;

inline std::istream& operator>>(std::istream& is, Vec3& t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void Vec3::make_unit_vector() {
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline Vec3 operator+(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline Vec3 operator/(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vec3 operator/(Vec3 v, float t) {
    return Vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v, float t) {
    return Vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const Vec3& v1, const Vec3& v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline Vec3 cross(const Vec3& v1, const Vec3& v2) {
    return Vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}


__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
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

__host__ __device__ inline Vec3& Vec3::operator-=(const Vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) {
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline Vec3 unit_vector(Vec3 v) {
    return v / v.length();
}

// 3d geometry
#define RANDVEC3 Vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ Vec3 random_in_unit_sphere(curandState* local_rand_state) {
    Vec3 p;
    do {
        p = 2.0f * RANDVEC3 - Vec3(1, 1, 1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__device__ Vec3 random_unit_vector(curandState* local_rand_state) {
    return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ Vec3 reflect(const Vec3& v, const Vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__host__ __device__ inline bool Vec3::near_zero() const {
    // Return true if the vector is close to zero in all dimensions.
    const auto s = 1e-8;
    return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
}