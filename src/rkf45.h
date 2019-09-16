#include <iostream>
#include <Eigen/Dense>

#ifdef USE_GPU
#define GPU_ENABLED_FUNC __host__ __device__
#else
#define GPU_ENABLED_FUNC
#endif

template<typename ODE>
class RungeKuttaFehleberg45
{
public:
    using ode_type = ODE;
    using state_type = typename ODE::state_type;
    using value_type = typename ODE::value_type;

protected:
    using ATableauType = Eigen::Array<typename ODE::value_type, 5,5>;
    using BTableauType = Eigen::Array<typename ODE::value_type, 2,6>;
    using CTableauType = Eigen::Array<typename ODE::value_type, 5,1>;


    ATableauType a_coefficients;
    BTableauType b_coefficients;
    CTableauType c_coefficients;

    value_type error_tolerance;
    value_type step_size;

public:
    GPU_ENABLED_FUNC RungeKuttaFehleberg45(const value_type& initial_tolerance, const value_type& initial_step_size) : error_tolerance(initial_tolerance), step_size(initial_step_size)
    {
        // initialize coefficients
        this->a_coefficients.setZero();
        this->b_coefficients.setZero();
        this->c_coefficients.setZero();

        this->a_coefficients(0,0) = 0.25;

        this->a_coefficients(1,0) = 3.0/32.0;
        this->a_coefficients(1,1) = 9.0/32.0;

        this->a_coefficients(2,0) = 1932.0/2197.0;
        this->a_coefficients(2,1) = -7200.0/2197.0;
        this->a_coefficients(2,2) = 7296.0/2197.0;

        this->a_coefficients(3,0) = 439.0/216.0;
        this->a_coefficients(3,1) = -8.0;
        this->a_coefficients(3,2) = 3680.0/513.0;
        this->a_coefficients(3,3) = -845.0/4104.0;

        this->a_coefficients(4,0) = -8.0/27.0;
        this->a_coefficients(4,1) = 2.0;
        this->a_coefficients(4,2) = -3544.0/2565.0;
        this->a_coefficients(4,3) = 1859.0/4104.0;
        this->a_coefficients(4,4) = -11.0/40.0;

        this->b_coefficients(0,0) = 16.0/135.0;
        this->b_coefficients(0,2) = 6656.0/12825.0;
        this->b_coefficients(0,3) = 28561.0/56430.0;
        this->b_coefficients(0,4) = -9.0/50.0;
        this->b_coefficients(0,5) = 2.0/55.0;

        this->b_coefficients(1,0) = 25.0/16.0;
        this->b_coefficients(1,2) = 1408.0/2565.0;
        this->b_coefficients(1,3) = 2197.0/4104.0;
        this->b_coefficients(1,4) = -1.0/5.0;

        this->c_coefficients(0) = 0.25;
        this->c_coefficients(1) = 3.0/8.0;
        this->c_coefficients(2) = 12.0/13.0;
        this->c_coefficients(3) = 1.0;
        this->c_coefficients(4) = 0.5;
    }


    GPU_ENABLED_FUNC void step(value_type& t, state_type& x, const ODE& ode)
    {
        bool accepted = false;
        state_type working_x = x;
        value_type working_t = t;

        // calculating forst approximation k1, which is independent of the step size
        state_type k1 = ode(t,x);
        while(!accepted)
        {
            // integrate
            working_x = x;
            working_t = t;
            value_type error = this->integrate(k1, working_x, working_t);

            // do we accept the approximation?
            if(error > this->error_tolerance)
            {
                accepted = true;
            }

            // adapt step size
            value_type alpha = pow(this->error_tolerance/error,1.0/6.0);
            this->step_size *= alpha;
        }
    }

protected:

    GPU_ENABLED_FUNC value_type integrate(const state_type& k1, value_type& t, state_type& x, const ODE& ode)
    {
        state_type tmp_x = x + this->step_size*this->a_coefficients(0,0)*k1;
        state_type k2 = ode(t+this->c_coefficients(0)*this->step_size, tmp_x);

        tmp_x = x + this->step_size*(this->a_coefficients(1,0)*k1 + this->a_coefficients(1,1)*k2);
        state_type k3 = ode(t+this->c_coefficients(1)*this->step_size, tmp_x);

        tmp_x = x + this->step_size*(this->a_coefficients(2,0)*k1 + this->a_coefficients(2,1)*k2 + this->a_coefficients(2,2)*k3);
        state_type k4 = ode(t+this->c_coefficients(2)*this->step_size, tmp_x);

        tmp_x = x + this->step_size*(this->a_coefficients(3,0)*k1 + this->a_coefficients(3,1)*k2 + this->a_coefficients(3,2)*k3 + this->a_coefficients(3,3)*k4);
        state_type k5 = ode(t+this->c_coefficients(3)*this->step_size, tmp_x);

        tmp_x = x + this->step_size*(this->a_coefficients(4,0)*k1 + this->a_coefficients(4,1)*k2 + this->a_coefficients(4,2)*k3 + this->a_coefficients(4,3)*k4 + this->a_coefficients(4,4)*k5);
        state_type k6 = ode(t+this->c_coefficients(4)*this->step_size, tmp_x);

        state_type x_rk4 = x + this->step_size*(this->b_coefficients(1,0)*k1+this->b_coefficients(1,2)*k3+this->b_coefficients(1,3)*k4+this->b_coefficients(1,4)*k5);
        x = x + this->step_size*(this->b_coefficients(0,0)*k1+this->b_coefficients(0,2)*k3+this->b_coefficients(0,3)*k4+this->b_coefficients(0,4)*k5+this->b_coefficients(0,5)*k6);

        return x - x_rk4;
    }
};
