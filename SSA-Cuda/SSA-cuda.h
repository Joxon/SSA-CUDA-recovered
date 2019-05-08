#pragma once

#include <device_functions.h>
#include <device_launch_parameters.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// thrust::device_vector can not be used in __global__ kernel
#include <thrust\execution_policy.h>
#include <thrust\device_vector.h>
#include <thrust\transform_reduce.h>
#include <thrust\functional.h>
#include <thrust\copy.h>
#include <thrust\extrema.h>

#include "math_constants.h"
#include "helper_cuda.h"

#include "SSA.h"

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif
// Now launch your kernel using the appropriate macro:
//kernel KERNEL_ARGS2(dim3(nBlockCount), dim3(nThreadCount)) (param1);

#define MAX_THREADS_PER_MP 2048
#define MAX_THREADS_PER_BLOCK 1024
#define THREADS_PER_BLOCK 64
#define DIM_THREADS (dim3 (THREADS_PER_BLOCK, 1))
#define DIM_BLOCKS(probSize) (dim3 (((probSize / THREADS_PER_BLOCK) + 1), 1))

//f1(z)=sum(i=1,n)(zi^2)

//f2(z)=sum(i=1,n)(|zi|) + product(i=1,n)(|zi|)

//f3(z)=z1^2 + 10^6*sum(i=2,n)(zi^2)

//f4(z)=10^6*z1^2 + sum(i=2,n)(zi^2)

//f5(z)=sum(i=1,n)(i*zi^4) + rand() (0<rand()<1)

//f6(z)=sum(i=1,n)(zi^2-10cos(2*pi*zi)+10)

//f7(z)=-20*exp(-0.2*sqrt((1/n)*sum(i=1,n)(zi^2))) - exp((1/n)*sum(i=1,n)(cos(2*pi*zi))) + 20 + e

//f8(z)=(1/4000)*sum(i=1,n)(zi^2) - product(i=1,n)(cos(zi/sqrt(i))) + 1

//f9(z)=sum(i=1,n-1)((100*(z(i+1)-zi^2))^2 + (zi-1)^2)

//f10(z)=(sin(pi*y1))^2 +
//       sum(i=1,n-1)((yi-1)^2*(1+10*((sin(y(i+1)))^2))) +
//       (yn-1)^2*(1+sin(2*pi*yn)^2)
//yi=1+(1/4)*(zi+1)

//f11(z)=(1/10)*(sin(3*pi*z1)^2 + sum(i=1,n-1)((zi-1)^2*(1+sin(3*pi*z(i+1))^2)) + (zn-1)^2*(1+sin(2*pi*zn)^2)) + sum(i=1,n)u(zi,5,100,4)
//u(zi,a,k,m) = k*(zi-a)^m, if(zi>a)
//u(zi,a,k,m) = 0, if(-a<=zi<=a)
//u(zi,a,k,m) = k*(-zi-a)^m, if(zi<-a)

//f12(z)=sum(i=1,n-1)(g(zi,z(i+1))) + g(zn,z1)
//g(x,y)=0.5 + ((sin(sqrt(x^2+y^2)))^2-0.5)/((1+0.001*(x^2+y^2))^2)

//f13(z)=418.9828872724338*n - sum(i=1,n)(zi*sin(sqrt(|zi|)))

//f14(z)=((1/(n-1))*sum(i=1,n-1)(sqrt(yi)+sin(50*yi^0.2)*sqrt(yi))^2
//yi=sqrt(zi^2+z(i+1)^2)

//f15(z)=min(sum(i=1,n)((zi-u1)^2),d*n+s*sum(i=1,n)((zi-u2)^2)+10*sum(i=1,n)(1-cos(2*pi*(zi-u1))))
//d=1
//s=1-1/(2*sqrt(n)-8.2)
//u1=2.5
//u2=-sqrt((u1^2-1)/s)

#define F1

template <typename T>
__device__ inline T u(T zi, T a, T k, T m)
{
    if (zi > a) { return k * pow(zi - a, m); }
    else if (zi < -a) { return k * pow(-zi - a, m); }
    else { return 0; }
}

template <typename T>
__device__ inline T g(T x, T y)
{
    return 0.5 +
        (pow(sin(sqrt(x * x + y * y)), 2) - 0.5) /
        pow((1 + 0.001*(x * x + y * y)), 2);
}

template <typename T>
__device__ inline T y14(T zi, T zi1)
{
    return sqrt(zi * zi + zi1 * zi1);
}

template <typename T>
struct f6_functor
{
    __host__ __device__
        T operator()(const T& x) const
    {
        return x * x - 10 * cos(2 * CUDART_PI_F * x) + 10;
    }
};

template <typename T>
struct f7_functor
{
    __host__ __device__
        T operator()(const T& x) const
    {
        return cos(2 * CUDART_PI_F * x);
    }
};

template <typename T>
struct f8_functor
{
    __host__ __device__
        T operator()(const T& x) const
    {
        return cos(2 * CUDART_PI_F * x);
    }
};

template <typename T>
struct f13_functor
{
    __host__ __device__
        T operator()(const T& x) const
    {
        return x * sin(sqrt(abs(x)));
    }
};

template <typename T>
struct f15_functor
{
    const T u1;

    __host__ __device__
        f15_functor(T u) : u1(u) {}

    __host__ __device__
        T operator()(const T& x) const
    {
        return 1 - cos(2 * CUDART_PI_F*(x - u1));
    }
};

template <typename T>
struct square_functor
{
    __host__ __device__
        T operator()(const T& x) const
    {
        return x * x;
    }
};

template <typename T>
struct abs_functor
{
    __host__ __device__
        T operator()(const T& x) const
    {
        return x >= T(0) ? x : -x;
    }
};

template <typename T>
struct variance_functor
{
    const T mean;

    __host__ __device__
        variance_functor(T m) : mean(m) {}

    __host__ __device__
        T operator()(const T& data) const
    {
        float x = (data - mean);
        return x * x;
    }
};

__global__ void getSquareKernel(float *dSqrt,
                                float *dData,
                                size_t dataSize);

__global__ void getSumKernel(float *dSum,
                             float *dData,
                             size_t dataSize);

__global__ void curandInitKernel(unsigned long long seed,
                                 curandState *devState,
                                 size_t devStateSize);

__global__ void randomWalkKernel(float ** devVibPosSol,
                                 float * devPrevMove,
                                 float * devPosSol,
                                 unsigned * devDimMask,
                                 float * devTgtVibPosSol,
                                 curandState * devStates,
                                 size_t popSize,
                                 size_t probDim);

__global__ void getAllStdDev(float *dStdDev,
                             float *dSol,
                             size_t probDim,
                             size_t popSize);

__global__ void chooseVibrationKernel(int *dMaxIndex,
                                      float *dMaxIntensity,
                                      float ** dVibPosSol,
                                      float ** dSpdTgtPosSol,
                                      float * dVibInt,
                                      float ** dDist,
                                      size_t popSize,
                                      size_t probDim,
                                      float attenuation_factor);

__global__ void getAllFitnesses(float * dFitness,
                                float * dPopPosSol,
                                size_t popSize,
                                size_t probDim);

__global__ void getAllDistances(float * dDist1D,
                                float ** dPopPosSol2,
                                size_t popSize,
                                size_t probDim);

extern curandState *dStates;
