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
	virtual thrust::device_vector<T> compute(const thrust::device_vector<T>& in) = 0;

	/**
	 * @brief compute on CPU
	 */
	virtual thrust::host_vector<T> compute(const thrust::host_vector<T>& in) = 0;
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
	template<typename T>
	T square(const T& t)
	{
		T out(t.size());
		thrust::transform(t.begin(), t.end(), out.begin(), square_functor());
		return out;
	}

	/**
	 * @brief compute on GPU
	 */
	virtual thrust::device_vector<T> compute(const thrust::device_vector<T>& in)
	{
		std::cout << "Calculating on GPU\n";
		return this->square(in);
	}

	/**
	 * @brief compute on CPU
	 */
	virtual thrust::host_vector<T> compute(const thrust::host_vector<T>& in)
	{
		std::cout << "Calculating on CPU\n";
		return this->square(in);
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
	template<typename T>
	T add_one(const T& t)
	{
		T out(t.size());
		thrust::transform(t.begin(), t.end(), out.begin(), addone_functor());
		return out;
	}

	/**
	 * @brief compute on GPU
	 */
	virtual thrust::device_vector<T> compute(const thrust::device_vector<T>& in)
	{
		std::cout << "Calculating on GPU\n";
		return this->add_one(in);
	}

	/**
	 * @brief compute on CPU
	 */
	virtual thrust::host_vector<T> compute(const thrust::host_vector<T>& in)
	{
		std::cout << "Calculating on CPU\n";
		return this->add_one(in);
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
	template<typename T>
	T add_arb(const T& t)
	{
		T out(t.size());
		thrust::transform(t.begin(), t.end(), out.begin(), addarb_functor(this->v));
		return out;
	}

	/**
	 * @brief compute on GPU
	 */
	virtual thrust::device_vector<T> compute(const thrust::device_vector<T>& in)
	{
		std::cout << "Calculating on GPU\n";
		return this->add_arb(in);
	}

	/**
	 * @brief compute on CPU
	 */
	virtual thrust::host_vector<T> compute(const thrust::host_vector<T>& in)
	{
		std::cout << "Calculating on CPU\n";
		return this->add_arb(in);
	}

	T v;
};

template<typename DataType>
void time_execution(const DataType& data, Interface<typename DataType::value_type>& interface)
{
	auto start = std::chrono::high_resolution_clock::now();
	interface.compute(data);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "Execution took " << duration.count() << " microseconds\n";
}

int main(void)
{
  const size_t N = 40;
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);

  // allocate storage for points
  thrust::device_vector<float> data_d(N);
  thrust::host_vector<float> data_h(N);

  // generate some random data
  for(size_t i = 0; i < N; i++)
  {
      data_d[i] = u01(rng);
      data_h[i] = u01(rng);
  }

  // AddOne
  time_execution(data_d, AddOne());
  time_execution(data_h, AddOne());

  // Square
  time_execution(data_d, Square());
  time_execution(data_h, Square());

  // AddArb
  time_execution(data_d, AddArb(5));
  time_execution(data_h, AddArb(5));

  return 0;
}
