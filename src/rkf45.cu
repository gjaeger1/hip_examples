
#define USE_GPU
#include "rkf45.h"
#include <hip/hip_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>
#include <thrust/pair.h>
#include <thrust/random.h>
#include <thrust/extrema.h>

/**
 * @brief A very simple ODE that we want to simulate here.
 */
template<typename T>
struct ODE
{
    using value_type = T;
    using state_type = T;

    __host__ __device__ state_type operator()(const value_type& t, const state_type& x) const
    {return 5*x-3;}
};

/**
 * @brief Simulator.
 *
 * The template parameter 'Integrator' is the numerical solver with the corresponding ODE system configured. The simulator is the functor the will be used by thrust::transform
 */
template<typename Integrator>
struct Simulator
{
    using ode_type = typename Integrator::ode_type;
    using value_type = typename Integrator::value_type;
    using state_type = typename Integrator::state_type;

    value_type t_start;
    value_type t_end;
    value_type dt;
    value_type tolerance;

    /**
     * @brief Constructor that takes typical simulation parameter as input arguments
     */
    __host__ __device__
    Simulator(value_type& end, value_type& dt, value_type& tolerance): t_start(0.0), t_end(end), dt(dt), tolerance(tolerance) {};

    /**
     * @brief operator() overload so that we can use the Simulator as a functor. Here is where the ODE will be solved/integrated using the 'Integrator'.
     */
    template<typename Tuple>
    __host__ __device__ state_type operator()(Tuple tupel)
    {
        // extract inputs from tupel
        ode_type ode = thrust::get<1>(tupel);
        state_type working_x = thrust::get<0>(tupel);

        // construct an integrator, e.g. RungeKuttaFehlberg45
        Integrator i(this->tolerance,this->dt);
        
        // initialize time
        value_type t = this->t_start;

        // loop until we reached the specified ending time.
        while(t <= this->t_end)
        {
            i.step(t, working_x, ode);
        }
        
        // return state at the end of simulation
        return working_x;
    }
};

int main(int argc, char** argv)
{
    using value_type = double;
    using state_type = double;
    using ode_type = ODE<value_type>;

    const size_t N = 4000; // number of parallel simulations
    value_type t_end = 0.5;
    value_type dt = 0.1;
    value_type tolerance = 0.0001;

    thrust::device_vector<state_type> states_d(N);
    thrust::device_vector<state_type> out_states_d(N);
    thrust::device_vector<ode_type> odes_d(N);

    thrust::host_vector<state_type> states_h(N);
    thrust::host_vector<state_type> out_states_h(N);
    thrust::host_vector<ode_type> odes_h(N);

    // generate some random data
    std::cout << "Generating " << N << " random values\n";
    thrust::default_random_engine rng;
    thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
    for(size_t i = 0; i < N; i++)
    {
      value_type rnd = u01(rng);
      states_h[i] = rnd;
    }

    states_d = states_h;
    odes_d = odes_h;
    
    // host
	std::cout << "Calculating on host...\n";
    auto start = std::chrono::high_resolution_clock::now();
	
    thrust::transform(
		    thrust::make_zip_iterator( thrust::make_tuple( states_h.begin() , odes_h.begin() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( states_h.end() , odes_h.end() ) ) ,
		    out_states_h.begin(),
		    Simulator<RungeKuttaFehleberg45<ode_type>>(t_end, dt, tolerance));
    auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "Execution took " << duration.count() << " microseconds\n";

    // device
    std::cout << "Calculating on device...\n";
    start = std::chrono::high_resolution_clock::now();
	thrust::transform(
		    thrust::make_zip_iterator( thrust::make_tuple( states_d.begin() , odes_d.begin() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( states_d.end() , odes_d.end() ) ) ,
		    out_states_d.begin(),
		    Simulator<RungeKuttaFehleberg45<ode_type>>(t_end, dt, tolerance));
	thrust::host_vector<state_type> out_states_h_d = out_states_d;
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	std::cout << "Execution took " << duration.count() << " microseconds\n";
    

	
	
	// compare data
	std::cout << "Checking results...\n";
	for(std::size_t i = 0; i < out_states_d.size(); i++)
	{
	    if(std::abs(out_states_h_d[i] - out_states_h[i]) > 0.01)
	    {
	        std::cout << "State of " << i << "-th simulation was " << out_states_h_d[i] << " on device and " << out_states_h[i] << " on host!\n";
	        return -1;
	    }
	}

    return 0;
}
