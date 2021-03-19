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

class Dielectric : public Material {
public:
    __device__ Dielectric(float index_of_refraction) : ir(index_of_refraction) {}
    __device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, Color& attenuation, Ray& scattered, curandState* local_rand_state) const {
        attenuation = Color(1.0, 1.0, 1.0);
        float refraction_ratio = rec.front_face ? (1.0 / ir) : ir;

        Vec3 unit_direction = unit_vector(r_in.direction());
        Vec3 refracted = refract(unit_direction, rec.normal, refraction_ratio);
        float cos_theta = float_min(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        Vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float(local_rand_state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        scattered = Ray(rec.p, direction);
        return true;

        //Vec3 outward_normal;
        //Vec3 reflected = reflect(r_in.direction(), rec.normal);
        //float ni_over_nt;
        //attenuation = Vec3(1.0, 1.0, 1.0);
        //Vec3 refracted;
        //float reflect_prob;
        //float cosine;
        //if (dot(r_in.direction(), rec.normal) > 0.0f) {
        //    outward_normal = -rec.normal;
        //    ni_over_nt = ir;
        //    cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
        //    cosine = sqrt(1.0f - ir * ir * (1 - cosine * cosine));
        //}
        //else {
        //    outward_normal = rec.normal;
        //    ni_over_nt = 1.0f / ir;
        //    cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        //}
        //if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
        //    reflect_prob = schlick(cosine, ir);
        //else
        //    reflect_prob = 1.0f;
        //if (curand_uniform(local_rand_state) < reflect_prob)
        //    scattered = Ray(rec.p, reflected);
        //else
        //    scattered = Ray(rec.p, refracted);
        //return true;
    }
public:
    float ir; // Index of Refraction
};