#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

__device__ __constant__ double MUB = 9.2740091523E-24;
__device__ __constant__ double HBAR = 1.054571817E-34;
__device__ __constant__ double GS = 2.0;

// Landau-Lifshitz torque.
//- 1/(1+α²) [ m x B +  α m x (m x B) ]
extern "C" __global__ void
lltorque2(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul,
          float time, float fixed_dt, float wc, float brms_x, float brms_y, float brms_z,
          float* __restrict__ rk_mx, float* __restrict__ rk_my, float* __restrict__ rk_mz, float* __restrict__ rk_tau, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};

        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);
        // float3 torque = gilb * (mxH + alpha * cross(m, mxH)); // LLG equation

        // Adding new time-dependant term to equations
        float3 brms = {brms_x , brms_y, brms_z};
        float3 mxBrms = cross(m, brms); // m x Brms

        float3 rk_m = {rk_mx[i], rk_my[i], rk_mz[i]};

        // Intergal from 0 to t
        float3 si_sum_total = sin(wc*(time - rk_tau[i])) * rk_m * fixed_dt;

        // Summatory for all cells
        // https://developer.download.nvidia.com/cg/dot.html
        float sum_final = 0.0;

        for (int cell = 0; cell < N; cell++) {
          sum_final += dot(si_sum_total, brms);
        }

        float constant_term = 1; //(float)(pow(GS,2)*pow(MUB,2))/(pow(HBAR,3)); // Constant value (gs^2*mub^2)/hbar^3

        float3 new_term = 2 * constant_term * mxBrms * sum_final; // LLG equation with full new time-dependant term to plug in equation

        float3 torque = (gilb * (mxH + alpha * cross(m, mxH))) - (new_term);

        // float3 torque = gilb * (mxH + alpha * cross(m, mxH)); // LLG equation

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}
