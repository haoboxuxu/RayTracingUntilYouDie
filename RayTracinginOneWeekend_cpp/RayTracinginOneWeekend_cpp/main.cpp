//
//  main.cpp
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/2/4.
//

#include <iostream>
#include <fstream>
#include "vec3.h"
using namespace std;

int main() {
    ofstream fout("/Users/haoboxuxu/Desktop/output.ppm");
    if (fout.fail()) {
        return -1;
    }
    
    const int image_width = 256;
    const int image_height = 256;
    
    fout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    
    for (int j = image_height-1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            auto r = double(i) / (image_width-1);
            auto g = double(j) / (image_height-1);
            auto b = 0.25;
            
            int ir = static_cast<int>(255.999 * r);
            int ig = static_cast<int>(255.999 * g);
            int ib = static_cast<int>(255.999 * b);
            
            // fout << ir << ' ' << ig << ' ' << ib << '\n';
            Vec3 color(double(i)/image_width, double(j)/image_height, 0.2);
            color.write_color(fout);
        }
    }
    
    fout.close();
    
    return 0;
}
