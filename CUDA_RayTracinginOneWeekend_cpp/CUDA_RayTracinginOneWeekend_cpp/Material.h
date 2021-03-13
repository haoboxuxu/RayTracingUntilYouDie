#pragma once

#include "Ray.h"
#include "Vec3.h"
#include "Hitable.h"
#define RANDVEC3 Vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

class Material {
public:
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* local_rand_state) const = 0;
};

class Lambertian : public Material {
public:
    __device__ Lambertian(const Vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Vec3& attenuation, Ray& scattered, curandState* local_rand_state) const {
       auto scatter_direction = rec.normal + random_unit_vector(local_rand_state);

        // Catch degenerate scatter direction
        if (scatter_direction.near_zero())
            scatter_direction = rec.normal;

        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
public:
    Vec3 albedo;
};

class Metal : public Material {
public:
    __device__ Metal(const Vec3& a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* local_rand_state) const {
        Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }

public:
    Color albedo;
    float fuzz;
};