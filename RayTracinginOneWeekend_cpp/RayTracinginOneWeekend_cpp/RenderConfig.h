//
//  RenderConfig.h
//  RayTracinginOneWeekend_cpp
//
//  Created by 徐浩博 on 2021/2/5.
//

#ifndef RenderConfig_h
#define RenderConfig_h

// Image

const auto aspect_ratio = 3.0 / 2.0;
const int image_width = 1080;
const int image_height = static_cast<int>(image_width / aspect_ratio);
const int samples_per_pixel = 500;
const int max_depth = 50;

#endif /* RenderConfig_h */
