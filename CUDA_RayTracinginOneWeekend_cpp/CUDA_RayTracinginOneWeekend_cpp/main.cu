//
//  main.cpp
//  CUDA_RayTracinginOneWeekend_cpp
//
//  Created by ÐìºÆ²© on 2021/3/8.
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include "Vec3.h"
#include "Ray.h"
#include "Sphere.h"
#include "Hitable.h"
#include "HitableList.h"
using namespace std;

// check cuda error
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__device__ bool hit_sphere(const Point3& center, float radius, const Ray& r) {
    Vec3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0f * dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - 4.0f * a * c;
    return (discriminant > 0.0f);
}

__device__ Vec3 color(const Ray& r, Hitable** world) {
    HitRecord rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f * Vec3(rec.normal.x() + 1.0f, rec.normal.y() + 1.0f, rec.normal.z() + 1.0f);
    } else {
        Vec3 unit_direction = unit_vector(r.direction());
        float t = 0.5f * (unit_direction.y() + 1.0f);
        return (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
    }
}

__global__ void render(Vec3* fb, int max_x, int max_y, Vec3 lower_left_corner, Vec3 horizontal, Vec3 vertical, Vec3 origin, Hitable** world) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    float u = float(i) / float(max_x);
    float v = float(j) / float(max_y);
    Ray r(origin, lower_left_corner + u * horizontal + v * vertical);
    fb[pixel_index] = color(r, world);
}

__global__ void create_world(Hitable** d_list, Hitable** d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(d_list) = new Sphere(Vec3(0, 0, -1), 0.5);
        *(d_list + 1) = new Sphere(Vec3(0, -100.5, -1), 100);
        *d_world = new HitableList(d_list, 2);
    }
}

__global__ void free_world(Hitable** d_list, Hitable** d_world) {
    delete* (d_list);
    delete* (d_list + 1);
    delete* d_world;
}

int main() {
    ofstream fout("C:\\Users\\haobo\\Desktop\\output.ppm");
    if (fout.fail()) {
        cout << "open file output.ppm failed" << endl;
        return -1;
    }

    // Image
    const auto aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    int tx = 8;
    int ty = 8;

    // World
    Hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(Hitable*)));
    Hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hitable*)));
    create_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Camera
    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    auto focal_length = 1.0;
    auto origin = Point3(0, 0, 0);
    auto horizontal = Vec3(viewport_width, 0, 0);
    auto vertical = Vec3(0, viewport_height, 0);
    auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - Vec3(0, 0, focal_length);

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    int num_pixels = image_width * image_height;
    size_t fb_size = 3 * num_pixels * sizeof(float);

    // allocate FB
    Vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render << <blocks, threads >> > (fb, image_width, image_height,
                                     lower_left_corner,
                                     horizontal,
                                     vertical,
                                     origin, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Output FB as Image
    fout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = image_height - 1; j >= 0; j--) {
        std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            int ir = int(255.99 * fb[pixel_index].r());
            int ig = int(255.99 * fb[pixel_index].g());
            int ib = int(255.99 * fb[pixel_index].b());
            fout << ir << " " << ig << " " << ib << "\n";
        }
    }

    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(fb));

    std::cerr << "\nDone.\n";

    fout.close();

    return 0;
}