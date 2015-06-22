#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include "matrix.h"

typedef unsigned int dev_index;
typedef unsigned int dev_size;

#define CUDA_TIME_OF(code) { \
		cudaEvent_t start; \
		cudaEvent_t stop; \
		cudaEventCreate(&start); \
		cudaEventCreate(&stop); \
		cudaEventRecord(start, 0); \
		code; \
		cudaEventRecord(stop, 0); \
		cudaEventSynchronize(stop); \
		float dt; \
		cudaEventElapsedTime(&dt, start, stop); \
		printf("Elapsed Time: %3.1f ms\n", dt); \
	}

#define DIV_UP(a, b) (a + b - 1) / (b)

inline int div_ceiling(const int a, const int b) {
	return (a + b - 1) / b;
}

inline __device__ int dev_div_ceiling(const int a, const int b) {
	return (a + b - 1) / b;
}

inline void assertSuccess() {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error %d: %s", cudaStatus, cudaGetErrorString(cudaStatus)); 
		exit(0);
	}
}

inline void assertSuccess(cudaError_t cudaStatus) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error %d: %s", cudaStatus, cudaGetErrorString(cudaStatus)); 
		exit(0);
	}
}

inline void assertSuccess(const char* message) {
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, message); 
		exit(0);
	}
}

inline void assertSuccess(cudaError_t cudaStatus, const char* message) {
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error %d: %s\nError Message: %s\n", cudaStatus, cudaGetErrorString(cudaStatus), message); 
		exit(0);
	}
}

template<typename decimal> class DeviceSystem {
public:
	decimal* arr;
	dev_size dim;
	DeviceSystem(dev_size n) {
		dim = n;
	}
	__device__ decimal* operator[](index i) {
		return (decimal*)((size)arr + sizeof(decimal)*i*(dim+1));
	}
	System<decimal>& toHostSystem() {
		System<decimal>* hostSystem = new System<decimal>(this->dim);
		size_t system_size = sizeof(float) * this->dim * (this->dim +1);
		assertSuccess(cudaMemcpy(hostSystem->array, this->arr, system_size, cudaMemcpyDeviceToHost), "cudaMemcpy failed to copy system form device to host.");
		return *hostSystem;
	}
};