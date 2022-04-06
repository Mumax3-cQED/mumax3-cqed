// MODIFIED INMA

#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

__device__ __constant__ double HBAR = 1.054571817E-34;

static __inline__ __device__ float3 operator*(const float3 &a, const float3 &b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// Landau-Lifshitz torque.
//- 1/(1+α²) [ m x B +  α m x (m x B) ]
extern "C" __global__ void
lltorque2time(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul,
          float delta_time, float wc, float brms_x, float brms_y, float brms_z,
          float* __restrict__ brmsi_x, float* __restrict__ brmsi_y, float* __restrict__ brmsi_z,
          float* __restrict__ rk_sin_mx, float* __restrict__ rk_sin_my, float* __restrict__ rk_sin_mz,
          float* __restrict__ rk_cos_mx, float* __restrict__ rk_cos_my, float* __restrict__ rk_cos_mz, float* __restrict__ ctime, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};

        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);
        // float3 torque = gilb * (mxH + alpha * cross(m, mxH)); // LLG equation

        // Adding new time-dependant term to equations
        brmsi_x[i] = brms_x;
        brmsi_y[i] = brms_y;
        brmsi_z[i] = brms_z;

        float3 brms = {brmsi_x[i] , brmsi_y[i], brmsi_z[i]};

        float3 mxBrms = cross(m, brms); // m x Brms

        float3 rk_sin_m = {rk_sin_mx[i], rk_sin_my[i], rk_sin_mz[i]};
        float3 rk_cos_m = {rk_cos_mx[i], rk_cos_my[i], rk_cos_mz[i]};

        // Intergal from 0 to t
        float3 si_sum_total = delta_time * ((cos(wc * ctime[i]) * rk_sin_m) - (sin(wc * ctime[i]) * rk_cos_m));

        // Summatory for all cells
        float3 sum_final = brms * si_sum_total;

        float hbar_const = (2 / HBAR);
        float3 new_term = (hbar_const * mxBrms * sum_final);

        float3 torque = (gilb * (mxH + alpha * cross(m, mxH))) - new_term; // LLG equation with full new time-dependant term to plug in equation

        // float3 torque = gilb * (mxH + alpha * cross(m, mxH)); // LLG equation

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}
