#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

#include <iostream>
#include <chrono>

/**
 * @brief exemplary interface class for transforming input data to output data
 */
template<typename T>
struct Interface
{
	/**
	 * @brief compute on GPU
	 */
	virtual void compute(thrust::device_vector<T>& out, const thrust::device_vector<T>& in) = 0;

	/**
	 * @brief compute on CPU
	 */
	virtual void compute(thrust::host_vector<T>& out, const thrust::host_vector<T>& in) = 0;
};

/**
 * @brief Square given input
 */
template<typename T>
struct Square: public Interface<T>
{
	/**
	 * @brief Helper Functor implementing 'square' functionality for GPU and CPU simultaneously
	 */
	struct square_functor
	{
		__host__ __device__ T operator()(const T& t) const {return t*t;}
	};

	/**
	 * @brief helper function to apply square
	 */
	template<typename VectorType>
	void square(VectorType& out, const VectorType& t)
	{
		thrust::transform(t.begin(), t.end(), out.begin(), square_functor());
	}

	/**
	 * @brief compute on GPU
	 */
	virtual void compute(thrust::device_vector<T>& out, const thrust::device_vector<T>& in)
	{
		std::cout << "Calculating on GPU\n";
		return this->square(out, in);
	}

	/**
	 * @brief compute on CPU
	 */
	virtual void compute(thrust::host_vector<T>& out, const thrust::host_vector<T>& in)
	{
		std::cout << "Calculating on CPU\n";
		return this->square(out, in);
	}
};

/**
 * @brief Add one to given input
 */
template<typename T>
struct AddOne: public Interface<T>
{
	/**
	 * @brief Helper Functor implementing 'square' functionality for GPU and CPU simultaneously
	 */
	struct addone_functor
	{
		__host__ __device__ T operator()(const T& t) const {return t+1;}
	};

	/**
	 * @brief helper function to apply square
	 */
	template<typename VectorType>
	void add_one(VectorType& out, const VectorType& t)
	{
		thrust::transform(t.begin(), t.end(), out.begin(), addone_functor());
	}

	/**
	 * @brief compute on GPU
	 */
	virtual void compute(thrust::device_vector<T>& out, const thrust::device_vector<T>& in)
	{
		std::cout << "Calculating on GPU\n";
		return this->add_one(out, in);
	}

	/**
	 * @brief compute on CPU
	 */
	virtual void compute(thrust::host_vector<T>& out, const thrust::host_vector<T>& in)
	{
		std::cout << "Calculating on CPU\n";
		return this->add_one(out, in);
	}
};

/**
 * @brief Add an arbitrary constant to given input
 */
template<typename T>
struct AddArb: public Interface<T>
{
	/**
	 * @brief Helper Functor implementing 'square' functionality for GPU and CPU simultaneously
	 */
	struct addarb_functor
	{
		__host__ __device__  addarb_functor(const T& t):v(t){};
		__host__ __device__ T operator()(const T& t) const {return t+this->v;}

		T v;
	};

	/**
	 * @brief Constructor taking the arbitrary constant to add as an input
	 */
	AddArb(const T& t):v(t){};

	/**
	 * @brief helper function to apply square
	 */
	template<typename VectorType>
	void add_arb(VectorType& out, const VectorType& t)
	{
		thrust::transform(t.begin(), t.end(), out.begin(), addarb_functor(this->v));
	}

	/**
	 * @brief compute on GPU
	 */
	virtual void compute(thrust::device_vector<T>& out, const thrust::device_vector<T>& in)
	{
		std::cout << "Calculating on GPU\n";
		return this->add_arb(out, in);
	}

	/**
	 * @brief compute on CPU
	 */
	virtual void compute(thrust::host_vector<T>& out, const thrust::host_vector<T>& in)
	{
		std::cout << "Calculating on CPU\n";
		return this->add_arb(out, in);
	}

	T v;
};

template<typename DataType>
void time_execution(DataType& out, const DataType& data, Interface<typename DataType::value_type>& interface)
{
	auto start = std::chrono::high_resolution_clock::now();
	interface.compute(out, data);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "Execution took " << duration.count() << " microseconds\n";
}

int main(void)
{
  const size_t N = 4000000;
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

  // allocate storage for points
  using value_type = float;
  thrust::device_vector<value_type> data_d(N);
  thrust::device_vector<value_type> out_d(N);
  
  thrust::host_vector<value_type> data_h(N);
  thrust::host_vector<value_type> out_h(N);

  // generate some random data
  std::cout << "Generating " << N << " random values\n";
  for(size_t i = 0; i < N; i++)
  {
  	  data_h[i] = u01(rng);
  }
  
  data_d = data_h;

  auto start = std::chrono::high_resolution_clock::now();
  // AddOne
  AddOne<value_type> f1;
  time_execution(out_d, data_d, f1);
  time_execution(out_h, data_h, f1);

  // Square
  Square<value_type> f2;
  time_execution(out_d, data_d, f2);
  time_execution(out_h, data_h, f2);

  // AddArb
  AddArb<value_type> f3(5);
  time_execution(out_d, data_d, f3);
  time_execution(out_h, data_h, f3);

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Overall execution took " << duration.count() << " microseconds\n";
  
  return 0;
}
