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
__global__ void matrixMul(const int* a, const int* b, int* c, int N) {
	// calculate global thread id - row and col
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	c[row * N + col] = 0;
	for (int k = 0; k < N; k++) {
		c[row * N + col] += a[row * N + k] * b[k * N + col];
	}
}

// check results
void verify_results(vector<int> &a, vector<int> &b, vector<int>& c, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			int tmp = 0;
			for (int k = 0; k < N; k++) {
				tmp += a[i * N + k] * b[k * N + j];
			}
			assert(tmp == c[i * N + j]);
		}
	}
	std::cout << "finished verify, no error\n";
}

int main() {
	// matrix size
	int N = 1 << 10;
	size_t bytes = N * N *sizeof(int);

	// vectors on cpu(host)
	vector<int> h_a(N * N);
	vector<int> h_b(N * N);
	vector<int> h_c(N * N);

	// init matrix
	generate(h_a.begin(), h_a.end(), []() { return rand() % 100; });
	generate(h_b.begin(), h_b.end(), []() { return rand() % 100; });
	
	// alloc memory on gpu(device)
	int* d_a, * d_b, * d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	// Copy data to the device
	cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

	// threads per cta
	int NUM_THREADS = 32;

	// cta per grid
	int NUM_BLOCKS = N / NUM_THREADS;

	dim3 threads(NUM_THREADS, NUM_THREADS);
	dim3 blocks(NUM_BLOCKS, NUM_BLOCKS);

	// run kernel on gpu
	matrixMul << <blocks, threads >> > (d_a, d_b, d_c, N);

	// Copy back to the host
	cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	// check
	verify_results(h_a, h_b, h_c, N);

	// free
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	std::cout << "COMPLETED SUCCESSFULLY\n";

	return 0;
}