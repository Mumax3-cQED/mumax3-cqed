#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

__device__ __constant__ double MUB = 9.2740091523E-24;
__device__ __constant__ double HBAR = 1.054571817E-34;
__device__ __constant__ double GS = 2.0;

__device__ float spinTorque(float calc_term, float mx_val, float my_val, float mz_val) {

  float sum_term = mx_val * calc_term + my_val * calc_term + mz_val * calc_term;
  return sum_term;
}

// Landau-Lifshitz torque.
//- 1/(1+α²) [ m x B +  α m x (m x B) ]
extern "C" __global__ void
lltorque2(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul, int N, float* __restrict__ dt, float* __restrict__ fixed_dt, float time, float wc, float brms_x, float brms_y, float brms_z, float* __restrict__ deltas) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {

        int idx = i;

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};

        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);

        // Adding new time-dependant term to equations
        float3 brms = {brms_x , brms_y, brms_z};
        float3 mxBrms = cross(m, brms); // m x Brms

        deltas[i] = dt[0];
        float fixdt_step = fixed_dt[0];

        __syncthreads();

        // Integral from 0 to T
        float si_sum_total = 0.0;

        for (int z = 0; z <= idx; z++) {

          float single_delta = deltas[z];

          if (single_delta > 0) {
              si_sum_total += spinTorque(sin(wc*(time - single_delta)), mx[z], my[z], mz[z]) * fixdt_step;
          }
        }

        float vectx = 0.0;
        float vecty = 0.0;
        float vectz = 0.0;

        // Summatory for all cells
        for (int z = 0; z <= idx; z++) {

          vectx += (brms.x * si_sum_total);
          vecty += (brms.y * si_sum_total);
          vectz += (brms.z * si_sum_total);
        }

        float vect_modulus = sqrt(pow(vectx, 2) + pow(vecty, 2) + pow(vectz, 2));
        float constant_term = (float)(pow(GS,2)*pow(MUB,2))/(pow(HBAR,3)); // Constant value (gs^2*mub^2)/hbar^3

        float3 new_term = 2 * constant_term * mxBrms * vect_modulus; // LLG equation with full new time-dependant term to plug in equation

        float3 torque = (gilb * (mxH + alpha * cross(m, mxH))) - (new_term);

        // float3 torque = gilb * (mxH + alpha * cross(m, mxH)); // LLG equation

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}
