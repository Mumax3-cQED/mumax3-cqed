#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include <iostream>
#include <stdio.h>
#include <math.h>

__device__ __constant__ double MUB = 9.2740091523E-24;
__device__ __constant__ double HBAR = 1.054571817E-34;
__device__ __constant__ double GS = 2.0;

//#define CONSTANT (powf(GS,2)*powf(MUB,2))/(powf(HBAR,3))

 // __device__ float d_si_sum_total = 0.0;
// __device__ int exec_threads = 0;


// inline __device__ __device__ float3 operator--(float3 a, float3 b)
// {
//     return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
// }

__device__ float spin_torque(float wc_frec, float mx_val, float my_val, float mz_val) {

  float sum_term = mx_val * wc_frec + my_val * wc_frec + mz_val * wc_frec;
  return sum_term;
}

// Landau-Lifshitz torque.
//- 1/(1+α²) [ m x B +  α m x (m x B) ]
extern "C" __global__ void
lltorque2(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul, int N, float dt, float fixed_dt, float time, float wc, float brms_x, float brms_y, float brms_z, float* __restrict__ deltas) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

    if (i < N) {

        int idx = i;
        deltas[i] = dt;

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};

        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);
        // float3 torque = gilb * (mxH + alpha * cross(m, mxH));

        //double h_bar = 1.054571817E-34; // h-bar planck value
        //double muB = 9.274009994E-24; // Bohr magneton
        //double gs = 2.0;
        float constant_term = (float)(powf(GS,2)*powf(MUB,2))/(powf(HBAR,3)); //  2.9334e+56;
        //float constant_term = (float)powf(gs,2)*pow(muB,2);//(float)(powf(gs,2)*powf(muB,2))/powf(h_bar,3);
      //  constant_term = fdividef(constant_term, powf(h_bar, 3));

        float3 brms = {brms_x , brms_y, brms_z};
        float3 mxBrms = cross(m, brms); // Si = m

        float si_sum_total = 0.0;

        for (int z = 0; z <= idx; z++) {
      //  for (float dtz = 0.0; dtz <= time; dtz+=dt) {
          si_sum_total += spin_torque(sin(wc*(time - deltas[z])), mx[z], my[z], mz[z]) * fixed_dt;
        }
        // si_sum_total = d_si_sum_total;

        // float value_sum = d_si_sum_total;
        //
        // d_si_sum_total = si_sum_total;
        // d_si_sum_total += (si_sum_total + val_sim_sum_total); //????
        //d_si_sum_total += val_sim_sum_total;
        //si_sum_total = d_si_sum_total;

        float full_term_zero = 0.0;
        float full_term_one = 0.0;
        float full_term_two = 0.0;

        for (int z = 0; z <= idx; z++) {
          full_term_zero += brms.x * si_sum_total;
          full_term_one +=  brms.y * si_sum_total;
          full_term_two +=  brms.z * si_sum_total;
        }

        // float3 items_term = {full_term_zero, full_term_one, full_term_two};
        float vect_modulus = sqrt(pow(full_term_zero, 2) + pow(full_term_one, 2) + pow(full_term_two, 2));

        float3 append_term = 2 * mxBrms * vect_modulus;
        // append_term = append_term * constant_term;

        float3 torque = (gilb * (mxH + alpha * cross(m, mxH))) - (append_term);

    //float3 torque = gilb * (mxH + alpha * cross(m, mxH));
        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;

    //    __syncthreads();
    }
}
