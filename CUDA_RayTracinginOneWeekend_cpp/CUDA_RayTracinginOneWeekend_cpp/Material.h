//
//  Material.h
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/2/5.
//

#ifndef Material_h
#define Material_h

#include "utilitys.h"

struct hit_record;

class Material {
public:
    virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const = 0;
};

class Lambertian : public Material {
public:
    Lambertian(const color& a) : albedo(a) {}
    
    virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {
        auto scatter_direction = rec.normal + random_unit_vector();
        
        // Catch degenerate scatter direction
        if (scatter_direction.near_zero()) {
            scatter_direction = rec.normal;
        }
        
        scattered = Ray(rec.p, scatter_direction);
        attenuation = albedo;
        return true;
    }
    
public:
    color albedo;
};

class Metal : public Material {
public:
    Metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}
    virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {
        Vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = Ray(rec.p, reflected + fuzz*random_in_unit_sphere());
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > 0);
    }
    
public:
    color albedo;
    double fuzz;
};

class Dielectric : public Material {
public:
    Dielectric(double index_of_refraction) : ir(index_of_refraction) {}
    
    virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered) const override {
        attenuation = color(1.0, 1.0, 1.0);
        double refraction_ratio = rec.front_face ? (1.0/ir) : ir;
        
        Vec3 unit_direction = unit_vector(r_in.direction());
        double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
        double sin_theta = sqrt(1.0 - cos_theta*cos_theta);
        
        bool cannot_refract = refraction_ratio * sin_theta > 1.0;
        Vec3 direction;
        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);
        
        scattered = Ray(rec.p, direction);
        return true;
    }
    
public:
    double ir; // Index of Refraction
    
private:
    static double reflectance(double cosine, double ref_idx) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1-ref_idx) / (1+ref_idx);
        r0 = r0*r0;
        return r0 + (1-r0)*pow((1 - cosine),5);
    }
};


#endif /* Material_h */
