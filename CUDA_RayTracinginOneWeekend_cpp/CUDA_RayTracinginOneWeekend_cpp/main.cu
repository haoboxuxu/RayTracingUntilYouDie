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

__global__ void render(Vec3* fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    fb[pixel_index] = Vec3(float(i) / max_x, float(j) / max_y, 0.2f);
}

int main() {
    ofstream fout("C:\\Users\\haobo\\Desktop\\output.ppm");
    if (fout.fail()) {
        cout << "open file output.ppm failed" << endl;
        return -1;
    }

    int image_width = 256;
    int image_height = 256;
    int tx = 8;
    int ty = 8;

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
    render << <blocks, threads >> > (fb, image_width, image_height);
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

    checkCudaErrors(cudaFree(fb));

    std::cerr << "\nDone.\n";

    fout.close();

    return 0;
}