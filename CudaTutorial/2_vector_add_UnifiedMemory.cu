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
	constexpr int N = 1 << 16;
	size_t bytes = sizeof(int) * N;

	// unified memory pointers
	int* a, * b, * c;

	// alloc memory for pointers
	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	// init nums in array
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	// threads per cta
	int NUM_THREADS = 1 << 10;

	// cta per grid
	int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

	// run kernel on gpu
	vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (a, b, c, N);

	// Wait for all previous operations before using values
	// We need this because we don't get the implicit synchronization of
	// cudaMemcpy like in the original example
	cudaDeviceSynchronize();

	// check
	verify_results(a, b, c, N);

	// free
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	std::cout << "COMPLETED SUCCESSFULLY\n";

	return 0;
}