#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include<vector>
using std::vector;

__device__ __constant__ float h_bar = 1.054571817e-34f; // h-bar planck value
__device__ __constant__ float muB = 9.274009994e-24f; // Bohr magneton
__device__ __constant__ float gs = 2.0f;

__device__ float si_sum_total = 0.0;
__device__ int exec_threads = 0;

__device__ float mul_vect(float scalar, float *vect, int v_len) {

  float res = 0.0f;

  for (int x = 0; x < v_len; x++) {
    res += vect[x] * scalar;
  }

  return res;
}

__device__ float spin_torque(float wc_frec, float *mi_tau_val, int vect_len) {

  float sum_term = mul_vect(sin(wc_frec), mi_tau_val, vect_len);
  return sum_term;
}

// Landau-Lifshitz torque.
//- 1/(1+α²) [ m x B +  α m x (m x B) ]
extern "C" __global__ void
lltorque2(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul, int N, float dt, float time, float wc, float brms_x, float brms_y, float brms_z) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {

        exec_threads += 1;

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};

        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);
        // float3 torque = gilb * (mxH + alpha * cross(m, mxH));

        float constant_term = ((pow(gs,2)*pow(muB,2))/pow(h_bar,3));

        float3 brms = {brms_x , brms_y, brms_z};
        float3 mxBrms = cross(m, brms); // Si = m

        // float *d_vect;
        float *h_vect;
        int n_lenght = 3;
        size_t bytes = n_lenght*sizeof(float);

        h_vect = (float*)malloc(bytes);

        h_vect[0] = mx[i];
        h_vect[1] = my[i];
        h_vect[2] = mz[i];

        si_sum_total += spin_torque(wc*(time - dt), h_vect, n_lenght) * dt;

        float3 full_term;

        for (int z = 0; z < exec_threads; z++) {
          full_term +=  brms * si_sum_total;
        }

        free(h_vect);

        float3 torque = gilb * (mxH + alpha * cross(m, mxH)) - 2 * constant_term * full_term;

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}
