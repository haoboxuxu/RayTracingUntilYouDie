//
//  texture.h
//  RayTracingingTheNextWeek_cpp
//
//  Created by 徐浩博 on 2021/5/10.
//

#ifndef texture_h
#define texture_h

#include "perlin.h"

class Texture {
public:
    virtual color value(double u, double v, const point3& p) const = 0;
};

class SolidColor : public Texture {
public:
    SolidColor() {}
    SolidColor(color c) : color_value(c) {}
    SolidColor(double red, double green, double blue) : SolidColor(color(red,green,blue)) {}
    
    color value(double u, double v, const point3 &p) const override {
        return color_value;
    }
    
private:
    color color_value;
};

class CheckerTexture : public Texture {
public:
    CheckerTexture() {}
    CheckerTexture(shared_ptr<Texture> _even, shared_ptr<Texture> _odd) : even(_even), odd(_odd) {}
    CheckerTexture(color c1, color c2) : even(make_shared<SolidColor>(c1)) , odd(make_shared<SolidColor>(c2)) {}
    
    color value(double u, double v, const point3 &p) const override {
        auto sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }
    
public:
    shared_ptr<Texture> odd;
    shared_ptr<Texture> even;
};

class NoiseTexture : public Texture {
public:
    NoiseTexture() {}
    color value(double u, double v, const point3 &p) const override {
        return color(1, 1, 1) * noise.noise(p);
    }
public:
    Perlin noise;
};


#endif /* texture_h */
