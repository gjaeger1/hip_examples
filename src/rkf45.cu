#include "rkf45.h"


template<typename T>
struct ODE
{
    using value_type = T;
    using state_type = T;

    __host__ __device__ state_type ode(const value_type& t, const state_type& x) const
    {return 5*x-3;}
};

template<typename Integrator>
struct Simulator
{
    using ode_type typename Integrator::ode_type;
    using value_type = typename Integrator::value_type;
    using state_type = typename Integrator::state_type;

    value_type t_start;
    value_type t_end;
    value_type dt;
    value_type tolerance;

    __host__ __device__
    Simulator(value_type& end, value_type& dt, value_type& tolerance): t_start(0.0), t_end(end), dt(dt), tolerance(tolerance) {};

    template<typename Tuple>
    __host__ __device__ state_type operator()(Tuple t)
    {
        ode_type ode = thrust::get<1>(t);
        state_type working_x = thrust::get<0>(t);

        Integrator i(this->tolerance,this->dt);
        value_type t = this->t_start;

        while(t <= this->t_end)
        {
            i.step(t, working_x, ode);
        }
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

int main(int argc, char** argv)
{
    using value_type = double;
    using state_type = double;
    using ode_type = ODE<value_type>;

    const size_t N = 4000000;
    value_type t_end = 10.0;
    value_type dt = 0.1;
    value_type tolerance = 0.0001;

    // allocate storage for points




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

    // device
    thrust::transform(
		    thrust::make_zip_iterator( thrust::make_tuple( states_d.begin() , odes_d.begin() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( states_d.end() , odes_d.end() ) ) ,
		    out_states_d.begin(),
		    Simulator<RungeKuttaFehleberg45<ode_type>>(t_end, dt, tolerance));

	// host
    thrust::transform(
		    thrust::make_zip_iterator( thrust::make_tuple( states_h.begin() , odes_h.begin() ) ),
            thrust::make_zip_iterator( thrust::make_tuple( states_h.end() , odes_h.end() ) ) ,
		    out_states_h.begin(),
		    Simulator<RungeKuttaFehleberg45<ode_type>>(t_end, dt, tolerance));

	// compare data
	thrust::host_vector<state_type> out_states_h_d = out_states_d;

	for(std::size_t i = 0; i < out_states_d.size(); i++)
	{
	    if(std::abs(out_states_h_d[i] - out_states_h[i]) > 0.01)
	    {
	        std::cout << "State of " << i "-th simulation was " << out_states_h_d[i] << " on device and " << out_states_h[i] << " on host!\n";
	        return -1;
	    }
	}

    return 0;
}
