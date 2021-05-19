//
//  utilitys.h
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/2/4.
//

#ifndef utilitys_h
#define utilitys_h

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <cstdlib>
#include <random>

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

// Utility Functions

inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

// Random Numbe

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double random_double(double min, double max) {
    // Returns a random real in [min,max).
    return min + (max-min)*random_double();
}

inline double clamp(double x, double min, double max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

inline int random_int(int min, int max) {
    // Returns a random integer in [min,max].
    return static_cast<int>(random_double(min, max+1));
}

// Common Headers

#include "Ray.h"
#include "Vec3.h"

#endif /* utilitys_h */
