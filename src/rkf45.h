#include <iostream>
#include <Eigen/Dense>

#ifdef USE_GPU
#define GPU_ENABLED_FUNC __host__ __device__
#else
#define GPU_ENABLED_FUNC
#endif

/**
 * @brief Implements the Runge-Kutta-Fehlberg 4/5 method with adaptive step size for solving ODEs.
 * 
 * See https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods and https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta%E2%80%93Fehlberg_method for details
 */
template<typename ODE>
class RungeKuttaFehleberg45
{
public:
    using ode_type = ODE;
    using state_type = typename ODE::state_type;
    using value_type = typename ODE::value_type;

protected:
    // types holding the RK45 coefficients
    using ATableauType = Eigen::Array<typename ODE::value_type, 5,5>;
    using BTableauType = Eigen::Array<typename ODE::value_type, 2,6>;
    using CTableauType = Eigen::Array<typename ODE::value_type, 5,1>;


    ATableauType a_coefficients;
    BTableauType b_coefficients;
    CTableauType c_coefficients;

    // error tolerance and step size to be adapted during simulation
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

    /**
     * @brief Integrates the given ode for a single step. Given time and state are updated accordingly.
     */
    GPU_ENABLED_FUNC void step(value_type& t, state_type& x, const ODE& ode)
    {
        bool accepted = false;
        state_type backup_x = x;
        value_type backup_t = t;

        // calculating forst approximation k1, which is independent of the step size
        state_type k1 = ode(t,x);
        while(!accepted)
        {
            // integrate
            x = backup_x;
            t = backup_t;
            value_type error = this->integrate(k1, t, x, ode);

            // do we accept the approximation?
            if(error > this->error_tolerance)
            {
                accepted = true;
            }

            // adapt step size
            value_type base = this->error_tolerance/error;
            
            //printf("Base: %f\n", base);
            
            if(isnan(base))
            {
                exit(-1);
                base = 10.0;
            }
            
            value_type alpha = pow(base,1.0/6.0);
            this->step_size *= alpha;
            
            //printf("Accepted: %d, alpha: %f, step size: %f\n", accepted, alpha, this->step_size);
        }
    }

protected:
    
    /**
     * @brief Apply RK 4/5 to get an approximate of the next step. The absolute difference between RK4 and RK5 is return as an local error estimate.
     */
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
        t += this->step_size;
        
        return abs(x - x_rk4);
    }
    
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
