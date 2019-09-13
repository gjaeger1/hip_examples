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
#include <vector>
#include <cmath>

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
 * Using CRTP pattern similarly to Eigen 3: see https://eigen.tuxfamily.org/dox/TopicInsideEigenExample.html, Section Construction of the sum expression, Paragraph about CRTP
 */
    
// Forward declarations so that we can access 'value_type' in derived classes (AddOne, AddArb, Square) from base class 'BaseFunctor'
template<typename Derived> struct type_traits;
template<typename T> struct AddOne;
template<typename T> struct AddArb;
template<typename T> struct Square;

// partial specialization of type_traits for all final functors


/**
 * @brief A Functor using CRTP pattern. 
 *
 * The Base class assumes that a derived class implements 'doWork' and exposes the function as a functor
 */
template<typename Derived>
struct BaseFunctor
{
	using T = typename type_traits<Derived>::value_type;
	__host__ __device__ T operator()(const T& t){return static_cast<Derived*>(this)->doWork(t);}
	__host__ __device__ T operator()(T&& t){return static_cast<Derived*>(this)->doWork(t);}
};

/**
 * @brief partial specialization of 'type_traits' for 'AddOne'
 */
template<typename T> struct type_traits<AddOne<T>>
{
	using value_type = T;
};

/**
 * @brief A simple functor that adds one
 */
template<typename T>
struct AddOne: public BaseFunctor<AddOne<T>>
{	
	using value_type = T;
	__host__ __device__ T doWork(const T& t){return t+1;}
	__host__ __device__ T doWork(T&& t){return t+1;}
};

/**
 * @brief partial specialization of 'type_traits' for 'Square'
 */
template<typename T> struct type_traits<Square<T>>
{
	using value_type = T;
};

/**
 * @brief A simple functor that squares the given value
 */
template<typename T>
struct Square: public BaseFunctor<Square<T>>
{
	using value_type = T;
	__host__ __device__ T doWork(const T& t){return t*t;}
	__host__ __device__ T doWork(T&& t){return t*t;}
};

/**
 * @brief partial specialization of 'type_traits' for 'AddArb'
 */
template<typename T> struct type_traits<AddArb<T>>
{
	using value_type = T;
};

/**
 * @brief A simple functor that adds an arbitrary value
 */
template<typename T>
struct AddArb: public BaseFunctor<AddArb<T>>
{
	using value_type = T;
	__host__ __device__ AddArb(const T& v): v(v) {};
	__host__ __device__ ~AddArb()=default;
	__host__ __device__ T doWork(const T& t){return t+this->v;}
	__host__ __device__ T doWork(T&& t){return t+this->v;}
	
	T v;
};

/**
 * @brief A composed unary functor that takes an input of type 'T' and applies three different functors before returning the transformed value.
 */
template<typename T, typename F1, typename F2, typename F3>
class ComposedUnaryFunctor
{
  public:
	__host__ __device__ ComposedUnaryFunctor(const F1& f1, const F2& f2, const F3& f3): f1(f1), f2(f2), f3(f3) {};
	__host__ __device__ ~ComposedUnaryFunctor()=default;
	
	__host__ __device__ T operator()(const T& in)
	{
		return f3(f2(f1(in)));
	}
  
  protected:
  	F1 f1;
  	F2 f2;
  	F3 f3;
};


/**
 * @brief The main kernel we want to execute. Note that we pass the struct that defined the actual calculation as a template parameter.
 */
template <typename T, typename Functor>
__global__ void main_kernel(T* C_d, const T* A_d, size_t N, Functor* c) 
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;
    
    for (size_t i = offset; i < N; i += stride) {
        C_d[i] = (*c)(A_d[i]);
    }
}

/**
 * @brief Testing function that executes the main kernel on GPU using the given functor and checks the results
 */
template <typename T>
class Tester
{
	const std::vector<T>& inputs;
	T* gpu_outputs;
	T* gpu_inputs;
	
  public:
  	/**
  	 * @brief Constructor that takes inputs for the Functor to test
  	 */
  	Tester(const std::vector<T>& inputs): inputs(inputs)
  	{
  		// allocate memory on gpu
  		std::size_t Nbytes = inputs.size()*sizeof(T);
  		std::cout << "Allocating " << 2*Nbytes / 1024.0 / 1024.0 << " MB on device\n";
    	CHECK(hipMalloc(&(this->gpu_outputs), Nbytes));
    	CHECK(hipMalloc(&(this->gpu_inputs), Nbytes));

		// copy inputs to device
		CHECK(hipMemcpy(this->gpu_inputs, this->inputs.data(), Nbytes, hipMemcpyHostToDevice));
  	}
  	
  	/**
  	 * @brief Destruct for releasing the memory allocated on GPU
  	 */
  	~Tester()
  	{
  		hipFree(this->gpu_inputs);
    	hipFree(this->gpu_outputs);
  	}
  	
  	/**
  	 * @brief Test whether or not the given Functor produces the same outputs (within a given range) on GPU as given as the first argument.
  	 */
  	template<typename Functor>
  	bool test(const std::vector<T>& expected_outputs, const Functor& f, const T& eps)
  	{
  		// copy 'Functor' to device
  		Functor* f_d;
		CHECK(hipMalloc((void**)&f_d, sizeof(Functor)));
		CHECK(hipMemcpy(f_d, &f, sizeof(Functor), hipMemcpyHostToDevice));
		
		const unsigned blocks = 512;
		const unsigned threadsPerBlock = 256;

		std::cout << "Launching Kernel...\n";
	   
	   	auto start = std::chrono::high_resolution_clock::now(); 
		// launch kernel
		hipLaunchKernelGGL(main_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, this->gpu_outputs, this->gpu_inputs, this->inputs.size(), f_d);
		hipDeviceSynchronize();
		
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);  
		
		std::cout << "Kernel execution took " << duration.count() << " microseconds" << std::endl;
		
		// get results from device
		std::vector<T> outputs(expected_outputs.size());
		CHECK(hipMemcpy(outputs.data(), this->gpu_outputs, outputs.size()*sizeof(T), hipMemcpyDeviceToHost));

		// check results
		bool result = true;
		for(std::size_t i = 0; i < outputs.size(); i++)
		{
			T diff = std::abs(outputs[i] - expected_outputs[i]);
			if (diff > eps) 
		    {
		    	std::cout << "Expected: " << expected_outputs[i] << " but was " << outputs[i] << std::endl;
		    	std::cout << "Absolute difference is " << diff  << " at " << i << std::endl;
		        result = false;
		    	break;
		    }
		}
	
		// free functor on device
		hipFree(f_d);
  		
  		return result;
  	}
};

int main(int argc, char** argv) 
{
	std::size_t N = 100000000;
    static int device = 0;
    using value_type = double;
    
    // prepare GPU
    CHECK(hipSetDevice(device));
    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, device /*deviceID*/));
    std::cout << "Running on device " << props.name << std::endl;
#ifdef __HIP_PLATFORM_HCC__
    std::cout << "Architecture on AMD GPU device is " << props.gcnArch << std::endl;
#endif


	// prepare test data
	std::vector<value_type> inputs(N);
	value_type cnt = 0;
	for(auto& v: inputs)
	{
		v = cnt;
		cnt += 0.5;
	}

	// create tester with input data
	Tester<value_type> tester(inputs);

	// first functor always adds one
	using F1 = ComposedUnaryFunctor<value_type, AddOne<value_type>, AddOne<value_type>, AddOne<value_type>>;
	AddOne<value_type> add_one;
	F1 f1(add_one,add_one,add_one);
	
	std::vector<value_type> targets(N);
	for(int i = 0; i < N; i++)
	{
		targets[i] = f1(inputs[i]);
	}
	
	if(tester.test(targets, f1, 0.00001))
	{
		std::cout << "PASSED!\n";
	}
	else
	{
		std::cout << "FAILED!\n";
	}
	
	// second functor is composed of all three functors
	using F2 = ComposedUnaryFunctor<value_type, AddOne<value_type>, Square<value_type>, AddArb<value_type>>;
	Square<value_type> square;
	AddArb<value_type> add_arb(2.0);
	F2 f2(add_one,square,add_arb);
	
	for(int i = 0; i < N; i++)
	{
		targets[i] = f2(inputs[i]);
	}
	
	if(tester.test(targets, f2, 0.00001))
	{
		std::cout << "PASSED!\n";
	}
	else
	{
		std::cout << "FAILED!\n";
	}
	
	return 0;
}
