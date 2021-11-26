#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include <iostream>
#include <stdio.h>

__device__ __constant__ float h_bar = 1.054571817e-34f; // h-bar planck value
__device__ __constant__ float muB = 9.274009994e-24f; // Bohr magneton
__device__ __constant__ float gs = 2.0f;

__device__ float dot(float scalar, float3 vect) {

  float res = 0.0f;//vect[0] * scalar + vect[1] *scalar + vect[2]*scalar;
  return res;
}

__device__ float spin_torque(float wc_frec, float3 mi_tau_val) {

  float sum_term = dot(sin(wc_frec), mi_tau_val);
  return sum_term;
}

// Landau-Lifshitz torque.
//- 1/(1+α²) [ m x B +  α m x (m x B) ]
extern "C" __global__ void
lltorque2(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul, int N, float dt, float wc, float si_sum, float brms_x, float brms_y, float brms_z) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};
        printf("%s\n", H);
        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);
        // float3 torque = gilb * (mxH + alpha * cross(m, mxH));

        float constant_term = ((pow(gs,2)*pow(muB,2))/pow(h_bar,3));
        float3 brms = {1.0f , 1.0f , 1.0f};
        float3 mxBrms = cross(m, brms); // Si = m

        //float spin_torque_val = spin_torque(wc, m);
        //si_sum += spin_torque(wc, m) * dt;
        // si_sum += sin(wc_frec) * m * dt;
        si_sum += 1;

        float3 torque = gilb * (mxH + alpha * cross(m, mxH)) - 2 * constant_term * mxBrms * si_sum;

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}
