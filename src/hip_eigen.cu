/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// check https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square for original sources

#include <stdio.h>
#include <Eigen/Dense>
#include <iostream>
#include <chrono> 
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

/**
 * @brief Square each element in vector. This function can be called only within a GPU Kernel.
 *
 * @param C_d array to write results to
 * @param A_d array whose elements we want to square
 * @param N Size of array C_d and A_d
 */
template<typename T>
__device__ void square(T* C_d, const T* A_d, const long int& N)
{
	Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>> A(A_d, N);
	Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>> C(C_d, N);

	C = A*A;
}

/**
 * @brief Main function of the kernel that we want to execute. It calculates which part of the array the current Thread should work on.
 *
 * @param C_d Output array
 * @param A_d Input array 
 * @param N Size of array C_d and A_d
 */
template<typename T>
__global__ void main_kernel(T* C_d, const T* A_d, std::size_t N)
{
    std::size_t num_threads = hipBlockDim_x * hipGridDim_x;
    std::size_t thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    std::size_t array_size = ceilf((float)N/(float)num_threads);
    std::size_t offset = thread_id * array_size;
    long int my_size = N - offset;
    if(my_size > array_size)
    	square(&C_d[offset], &A_d[offset], array_size);  
    else if (my_size > 0)
    	square(&C_d[offset], &A_d[offset], my_size);  
}

int main(int argc, char* argv[]) 
{
	// variables
    float *A_d;
    float *C_d;
    float *A_h;
    float *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);
    static int device = 0;
    
    // prepare GPU
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    printf("info: running on device %s\n", props.name);
#ifdef __HIP_PLATFORM_HCC__
    printf("info: architecture on AMD GPU device is: %d\n", props.gcnArch);
#endif
	
	// allocate memory on host side
    printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
    }

	// allocate memory on device side
    printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));
    CHECK(hipMalloc(&C_d, Nbytes));

	// copy  inputs from host to device
    printf("info: copy Host2Device\n");
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    printf("info: launch 'main_kernel' kernel\n");
	auto start = std::chrono::high_resolution_clock::now(); 
	
    hipLaunchKernelGGL(main_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);
	hipDeviceSynchronize();
	
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);  
	printf("info: kernel execution took %ld microseconds\n", duration.count());
    
    // copy results from device to host
    printf("info: copy Device2Host\n");
    CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

	// check
    printf("info: check result\n");
    for (size_t i = 0; i < N; i++) {
        if (C_h[i] != A_h[i] * A_h[i]) 
        {
            CHECK(hipErrorUnknown);
        }
    }
    printf("PASSED!\n");
    
    // free memory
    printf("info: free memory\n");
    hipFree(A_d);
    hipFree(C_d);
    free(A_h);
    free(C_h);
}
