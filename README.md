# HIP Examples

AMD's version of CUDA is called [ROCm](https://github.com/RadeonOpenCompute/ROCm). 
Most importantly, ROCm is extended by [HIP](https://github.com/ROCm-Developer-Tools/HIP), which is an abstraction layer to work on both AMD and Nvidia plattforms.
Thus, the idea is to implement only using HIP and compile it either for ROCm/HCC or CUDA/NVCC without having to change any code.

In this repository, you can find four small examples on how to use HIP. The examples were tested on a ROCm system (Radeaon RX Vega 64) using the rocm/rocm-terminal docker container as well as on the Nvidia Cluster DGX-2.

* hip_classes: An adapted version of [square](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square) which uses classes within the GPU kernel.
* hip_classes_crtp: Implementing the [CRTP](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) design pattern to dynamically construct GPU kernel based on template programming.
* hip_eigen: An adapted version of [square](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples/0_Intro/square) which uses the [Eigen 3 library](http://eigen.tuxfamily.org/index.php?title=Main_Page) within the GPU kernel. Note that the version stored in this repository tracks the development branch 'HIP_fixes' of the library. The default branch does not work with HIP, currently.
* thrust_bounding_box: One example from the [ROCm Thrust](https://github.com/ROCmSoftwarePlatform/Thrust/blob/master/examples/bounding_box.cu) library. It shows that we can use Thrust with CUDA and HIP without changes.
* boost_thrust_odeing: One of the examples given by the Boost library [odeint](https://www.boost.org/doc/libs/1_71_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/tutorial/using_cuda__or_openmp__tbb_______via_thrust.html) for how one can execute ODE simulations on a GPU. Although the example is given with respect to Thrust and CUDA, no changes are required to work with ROCm/HIP.

## Cloning

As ROCm/Thrust is included as a git submodule use `git clone --recursive` to clone the submodule as well.

## Dependencies

As Eigen 3 is a Mercurial repository, a copy of its 'HIP_fixes' branch is included.
Dependencies that need to be installed separately:

* boost
* cmake
* HIP
* CUDA (on Nvidia platforms)
* ROCm (on AMD platforms)

## Compilation

Execute the following commands in the root directory of the repository:

```
mkdir build
cd build
cmake ..
make
```

After a successfull compilation, you should see the individual executables in the 'build' folder.

## Docker

Provided that the Linux host has installed appropriate drivers, one can use [docker](https://docs.docker.com/) images to obtain ready-to-use coding environments.
[rocm/rocm-terminal](https://hub.docker.com/r/rocm/rocm-terminal) is a docker image based on Ubuntu 16.04.
[gjaeger1234/hip_cuda](https://hub.docker.com/r/gjaeger1234/hip_cuda) is a docker image integrating HIP with CUDA based on Ubuntu 18.04.
