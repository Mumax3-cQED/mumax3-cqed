// MODIFIED INMA

#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "constants.h"

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
        float3 brms = {brms_x, brms_y, brms_z};
        float3 mxBrms = cross(m, brms); // m x Brms

        // Summatory for all cells
        float cell_x, cell_y, cell_z = 0.0;

        for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
              ii < N; ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {

          float3 rk_sin_m = {rk_sin_mx[ii], rk_sin_my[ii], rk_sin_mz[ii]};
          float3 rk_cos_m = {rk_cos_mx[ii], rk_cos_my[ii], rk_cos_mz[ii]};

          float3 si_sum_total = ((cos(wc * ctime[ii]) * rk_sin_m) - (sin(wc * ctime[ii]) * rk_cos_m));

          __syncthreads();

          cell_x += si_sum_total.x;
          cell_y += si_sum_total.y;
          cell_z += si_sum_total.z;
        }

        float sum_temp_x = brms.x * delta_time * cell_x;
        float sum_temp_y = brms.y * delta_time * cell_y;
        float sum_temp_z = brms.z * delta_time * cell_z;

        float3 sum_final = {sum_temp_x, sum_temp_y, sum_temp_z};

        float spin_constant = 2 / HBAR; // debemos dividir entre gamma0 nuestro nuevo termino? parece que si
        float3 new_term = spin_constant * mxBrms * sum_final;

        float3 torque = gilb * (mxH + alpha * cross(m, mxH)) - new_term;  // LLG equation with full new time-dependant term to plug in equation

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}
