//
//  main.cpp
//  CUDA_RayTracinginOneWeekend_cpp
//
//  Created by ÐìºÆ²© on 2021/3/8.
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <fstream>
#include "Vec3.h"
#include "Ray.h"
#include "Sphere.h"
#include "Hitable.h"
#include "HitableList.h"
#include "Camera.h"
#include "config.h"
#include "Material.h"
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

__device__ Vec3 color(const Ray& r, Hitable** world, curandState* local_rand_state) {
    Ray cur_ray = r;
    Vec3 cur_attenuation = Vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < max_depth; i++) {
        HitRecord rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            Ray scattered;
            Vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return Vec3(0.0, 0.0, 0.0);
            }
        } else {
            Vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            Vec3 c = (1.0f - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return Vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(Vec3* fb, int max_x, int max_y, int samples_per_pixel, Camera **cam, Hitable** world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    Color pixel_color(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        Ray r = (*cam)->get_ray(u, v);
        pixel_color += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    pixel_color /= float(samples_per_pixel);
    pixel_color[0] = sqrt(pixel_color[0]);
    pixel_color[1] = sqrt(pixel_color[1]);
    pixel_color[2] = sqrt(pixel_color[2]);
    fb[pixel_index] = pixel_color;
}

__global__ void create_world(Hitable** d_list, Hitable** d_world, Camera** d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new Sphere(Vec3(0, 0, -1), 0.5, new Lambertian(Vec3(0.1, 0.2, 0.5)));
        d_list[1] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(0.8, 0.8, 0.0)));
        d_list[2] = new Sphere(Vec3(1, 0, -1), 0.5, new Metal(Vec3(0.8, 0.6, 0.2), 0.0));
        d_list[3] = new Sphere(Vec3(-1, 0, -1), 0.5, new Dielectric(1.5));
        d_list[4] = new Sphere(Vec3(-1, 0, -1), -0.45, new Dielectric(1.5));
        *d_world = new HitableList(d_list, 5);
        *d_camera = new Camera(Vec3(-2, 2, 1), Vec3(0, 0, -1), Vec3(0, 1, 0), 20.0, 16.0 / 9.0);
    }
}

__global__ void free_world(Hitable** d_list, Hitable** d_world, Camera** d_camera) {
    for (int i = 0; i < 5; i++) {
        delete ((Sphere*)d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete* d_world;
    delete* d_camera;
}

int main() {
    ofstream fout("C:\\Users\\haobo\\Desktop\\output.ppm");
    if (fout.fail()) {
        cout << "open file output.ppm failed" << endl;
        return -1;
    }

    // Image
    int tx = 8;
    int ty = 8;
    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(Vec3);

    // allocate FB
    Vec3* fb;
    checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

    // allocate random state
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    // World
    Hitable** d_list;
    checkCudaErrors(cudaMalloc((void**)&d_list, 2 * sizeof(Hitable*)));
    Hitable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(Hitable*)));
    // Camera
    Camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(Camera*)));
    create_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";


    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);
    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render << <blocks, threads >> > (fb, image_width, image_height, samples_per_pixel, d_camera, d_world, d_rand_state);
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

    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(fb));

    std::cerr << "\nDone.\n";

    fout.close();
    cudaDeviceReset();

    return 0;
}