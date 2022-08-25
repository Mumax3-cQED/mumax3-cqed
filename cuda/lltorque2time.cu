// MODIFIED INMA

#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "constants.h"
#include "stencil.h"

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
          float* __restrict__ rk_sin_mx, float* __restrict__ rk_sin_my, float* __restrict__ rk_sin_mz,
          float* __restrict__ rk_cos_mx, float* __restrict__ rk_cos_my, float* __restrict__ rk_cos_mz,  float* __restrict__ ctime,
          float* __restrict__ delta_time, float* __restrict__ brms_x, float* __restrict__ brms_y, float* brms_z, float* __restrict__ wc,
          int Nx, int Ny, int Nz, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};
        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);

        float3 new_term = make_float3(0.0, 0.0, 0.0);

        if (delta_time[i] > 0.0f) {

            // Summatory for all cells
            int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);
            int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
            int iz = ((blockIdx.z * blockDim.z) + threadIdx.z);

            if (ix >= Nx || iy >= Ny || iz >= Nz) {
                return;
            }

             // for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
             //       ii < N; ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {

            int ii = idx(ix, iy, iz);

            float3 rk_sin_m = {rk_sin_mx[ii], rk_sin_my[ii], rk_sin_mz[ii]};
            float3 rk_cos_m = {rk_cos_mx[ii], rk_cos_my[ii], rk_cos_mz[ii]};

            float3 brms = {brms_x[ii], brms_y[ii], brms_z[ii]};

            float3 sum_final = make_float3(0.0, 0.0, 0.0);
            sum_final += (brms * delta_time[ii] * ((cos(wc[ii] * ctime[ii]) * rk_sin_m) - (sin(wc[ii] * ctime[ii]) * rk_cos_m)));

             // }

            // Adding new time-dependant term to equation
            float3 mxBrms = cross(m, brms); // m x Brms

            float spin_constant = 2 / HBAR; // debemos dividir entre gamma0 nuestro nuevo termino? parece que si
            new_term = spin_constant * mxBrms * sum_final;
        }

        float3 torque = gilb * (mxH + alpha * cross(m, mxH)) - new_term;  // LLG equation with full new time-dependant term to plug in equation

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}
