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
#include "hip/hip_runtime.h"
#include <iostream>
#include <chrono> 

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
 * @brief Simple struct that multiplies a and b to store the result as its member 'r'
 */
template<typename T>
struct Result
{
	__host__ __device__ Result(const T& a, const T& b): r(a*b) {};
	__host__ __device__ ~Result()=default;
	T r;
};

/**
 * @brief Inheritence in GPU Kernels is not supported for virtual functions. However, 
 * we can still use the virtual keyword for functions that are defined only on the host side.
 */
template<typename T>
struct MyClass
{
	MyClass()=default;
	~MyClass()=default;
	virtual Result<T> doWorkHost(const T& a, const T& b) = 0;
};

/**
 * @brief The derived class may be part of a kernel and have a GPU-version of the actual calculations that have to be done.
 */
template<typename T>
struct Mul : public MyClass<T>
{
	__host__ __device__ Mul()=default;
	__host__ __device__ ~Mul()=default;
	__host__ Result<T> doWorkHost(const T& a, const T& b) override {return Result<T>(a,b);} 
	__device__ Result<T> doWorkGPU(const T& a, const T& b) {return Result<T>(a,b);} 
};


/**
 * @brief The main kernel we want to execute. Note that we pass the struct that defined the actual calculation as a template parameter.
 */
template <typename T, typename Functor>
__global__ void main_kernel(Result<T>* C_d, const T* A_d, size_t N, Functor* c) 
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;
    
    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = c->doWorkGPU(A_d[i], A_d[i]);
    }
}


int main(int argc, char* argv[]) {
    
    // variables
    // As we used __host__ and __device__ on the constructors/destructors of 'Result<T>' we can use the class on host and device side
    float *A_d;
    Result<float>* C_d;
    float *A_h;
    Result<float>* C_h;
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

	// allocate memory
    printf("info: allocate host mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    C_h = (Result<float>*)malloc(N * sizeof(Result<float>));
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess);
    // Fill with Phi + i
    for (size_t i = 0; i < N; i++) {
        A_h[i] = 1.618f + i;
    }

    printf("info: allocate device mem (%6.2f MB)\n", 2 * Nbytes / 1024.0 / 1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));
    CHECK(hipMalloc(&C_d, N * sizeof(Result<float>)));

	// copy inputs to device
    printf("info: copy Host2Device\n");
    CHECK(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

	// construct 'Functor'
	Mul<float>* obj;
	Mul<float> obj_h;
	CHECK(hipMalloc((void**)&obj, sizeof(Mul<float>)));
	// copy 'Functor' to device
	CHECK(hipMemcpy(obj, &obj_h, sizeof(Mul<float>), hipMemcpyHostToDevice));
	
    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    printf("info: launch 'main_kernel' kernel\n");
   
   	auto start = std::chrono::high_resolution_clock::now(); 
    // launch kernel
    hipLaunchKernelGGL(main_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N, obj);
    hipDeviceSynchronize();
	
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);  
	printf("info: kernel execution took %ld microseconds\n",duration.count());
    
    // get results from device
    printf("info: copy Device2Host\n");
    CHECK(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    printf("info: check result\n");
    for (size_t i = 0; i < N; i++) {
        if (C_h[i].r != A_h[i] * A_h[i]) 
        {
            CHECK(hipErrorUnknown);
        }
    }
    printf("PASSED!\n");
    
    // free memory
    printf("info: free memory\n");
    hipFree(A_d);
    hipFree(C_d);
    hipFree(obj);
    free(A_h);
    free(C_h);
}
