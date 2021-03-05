#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
using std::begin;
using std::copy;
using std::cout;
using std::end;
using std::generate;
using std::vector;

// cuda kernel for vec add
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
	// calculate global thread id
	int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (tid < N) c[tid] = a[tid] + b[tid];
}

// check results
void verify_results(int* a, int* b, int* c, int N) {
	for (int i = 0; i < N; i++) {
		assert(c[i] == a[i] + b[i]);
	}
	std::cout << "finished verify, no error\n";
}

int main() {
	// array size
	constexpr int N = 1 << 26;
	size_t bytes = sizeof(int) * N;

	// vectors cpu-side
	int* h_a, * h_b, * h_c;

	// alloc pinned memory
	cudaMallocHost(&h_a, bytes);
	cudaMallocHost(&h_b, bytes);
	cudaMallocHost(&h_c, bytes);

	// init nums in array
	for (int i = 0; i < N; i++) {
		h_a[i] = rand() % 100;
		h_b[i] = rand() % 100;
	}

	// alloc memory on device
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// copy cpu->gpu
	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	// threads per cta
	int NUM_THREADS = 1 << 10;

	// cta per grid
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	// run kernel on gpu
	vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, N);

	// copy sum gpu->cpu
	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	// check
	verify_results(h_a, h_b, h_c, N);

	// free
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "COMPLETED SUCCESSFULLY\n";

	return 0;
}