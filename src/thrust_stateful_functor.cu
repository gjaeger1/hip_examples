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
 * @brief Adds the given state to the given input and increases the state by one.
 *
 * Note that for stateful functors, it is necessary to have a separate state representation for each, host and device side.
 * A synchronization between both would be possible, but would most likely result in a loss of performance. Alternatively, one could
 * limit support for the functor to GPU or CPU exclusively.
 */
template<typename T>
struct StatefulFunctor: public Interface<T>
{
	/**
	 * @brief actual functor having a state, that is, how often the functor was already called.
	 */
	struct stateful_functor
	{
		T* cnt;

		__host__ stateful_functor(thrust::host_vector<T> const& cnts) : cnt(nullptr)
		{
			std::size_t thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
			T* raw_cnts = thrust::raw_pointer_cast(cnts.data());

			if(thread_id < cnts.size())
				this->cnt = raw_cnts + thread_id;
		}


		__device__ stateful_functor(thrust::device_vector<T> const& cnts) : cnt(nullptr)
		{
			std::size_t thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
			T* raw_cnts = thrust::raw_pointer_cast(cnts.data());

			if(thread_id < cnts.size())
				this->cnt = raw_cnts + thread_id;
		}

		__host__ __device__ T operator()(const T& t) const
		{
			T res = t + this->cnt;
			this->cnt += 1;
			return res;
		}
	};

	/**
	 * @brief actual states of our stateful_functor
	 */
	thrust::device_vector<T> states_d;
	thrust::host_vector<T> states_h;

	StatefulFunctor(const T& initial_state, const std::size_t& N): states_d(N, initial_state),states_h(N, initial_state) {};


	/**
	 * @brief compute on GPU
	 */
	virtual void compute(thrust::device_vector<T>& out, const thrust::device_vector<T>& in)
	{
		std::cout << "Calculating on GPU\n";
		thrust::transform(t.begin(), t.end(), out.begin(), stateful_functor(this->states_d));
	}

	/**
	 * @brief compute on CPU
	 */
	virtual void compute(thrust::host_vector<T>& out, const thrust::host_vector<T>& in)
	{
		std::cout << "Calculating on CPU\n";
		thrust::transform(t.begin(), t.end(), out.begin(), stateful_functor(this->states_h));
	}
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
      float rnd = u01(rng);
      data_h[i] = rnd;
  }

  data_d = data_h;

  auto start = std::chrono::high_resolution_clock::now();
  // Apply first time
  StatefulFunctor f(1.0, N);
  time_execution(out_d, data_d, f);
  time_execution(out_h, data_h, f);

  // check results
  for(size_t i = 0; i < N; i++)
  {
      if(out_h[i] != data_h[i] + 1.0)
      {
      	std::cout << "Host output at " << i << " should have been " << data_h[i] + 1.0 << " but was " << out_h[i] << "!\n";
      	return -1;
      }
  }

  // check results
  for(size_t i = 0; i < N; i++)
  {
      if(out_d[i] != data_d[i] + 1.0)
      {
      	std::cout << "Device output at " << i << " should have been " << data_d[i] + 1.0 << " but was " << out_d[i] << "!\n";
      	return -1;
      }
  }

  // Apply second time
  time_execution(out_d, data_d, f);
  time_execution(out_h, data_h, f);

  // check results
  for(size_t i = 0; i < N; i++)
  {
      if(out_h[i] != data_h[i] + 2.0)
      {
      	std::cout << "Host output at " << i << " should have been " << data_h[i] + 2.0 << " but was " << out_h[i] << "!\n";
      	return -1;
      }
  }

  // check results
  for(size_t i = 0; i < N; i++)
  {
      if(out_d[i] != data_d[i] + 2.0)
      {
      	std::cout << "Device output at " << i << " should have been " << data_d[i] + 2.0 << " but was " << out_d[i] << "!\n";
      	return -1;
      }
  }

  // Apply third time
  time_execution(out_d, data_d, f);
  time_execution(out_h, data_h, f);

  // check results
  for(size_t i = 0; i < N; i++)
  {
      if(out_h[i] != data_h[i] + 3.0)
      {
      	std::cout << "Host output at " << i << " should have been " << data_h[i] + 3.0 << " but was " << out_h[i] << "!\n";
      	return -1;
      }
  }

  // check results
  for(size_t i = 0; i < N; i++)
  {
      if(out_d[i] != data_d[i] + 3.0)
      {
      	std::cout << "Device output at " << i << " should have been " << data_d[i] + 3.0 << " but was " << out_d[i] << "!\n";
      	return -1;
      }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  std::cout << "Overall execution took " << duration.count() << " microseconds\n";

  return 0;
}
