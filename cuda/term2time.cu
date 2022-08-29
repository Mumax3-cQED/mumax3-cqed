// MODIFIED INMA

#include "amul.h"
#include "constants.h"
#include "stencil.h"
#include <stdio.h>

// static __inline__ __device__ float3 operator*(const float3 &a, const float3 &b) {
//   return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
// }

// Landau-Lifshitz torque.
//- 1/(1+α²) [ m x B +  α m x (m x B) ]
extern "C" __global__ void
term2time(float* __restrict__  new_term_x, float* __restrict__  new_term_y, float* __restrict__  new_term_z,
          float* __restrict__ rk_sin_mx, float* __restrict__ rk_sin_my, float* __restrict__ rk_sin_mz,
          float* __restrict__ rk_cos_mx, float* __restrict__ rk_cos_my, float* __restrict__ rk_cos_mz,
          float* __restrict__ delta_time, float* __restrict__ brms_x, float* __restrict__ brms_y, float* __restrict__ brms_z,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__ sumx, float* __restrict__ sumy, float* __restrict__ sumz,
          float ctimeWc, int N) {

          int ii =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

          if (ii < N) {

            float3 m = {mx[ii], my[ii], mz[ii]};

            float3 brms = {brms_x[ii], brms_y[ii], brms_z[ii]};

            sumx[ii] +=  (brms_x[ii] * delta_time[ii]) * (cos(ctimeWc) * rk_sin_mx[ii] - sin(ctimeWc) * rk_cos_mx[ii]);// funciona
            sumy[ii] +=  (brms_y[ii] * delta_time[ii]) * (cos(ctimeWc) * rk_sin_my[ii] - sin(ctimeWc) * rk_cos_my[ii]);
            sumz[ii] +=  (brms_z[ii] * delta_time[ii]) * (cos(ctimeWc) * rk_sin_mz[ii] - sin(ctimeWc) * rk_cos_mz[ii]);

            // Creating new time-dependant term
            float3 mxBrms = cross(m, brms); // m x Brms
            float spin_constant = 2 / HBAR; // debemos dividir entre gamma0 nuestro nuevo termino? parece que si

            float final_x = spin_constant * mxBrms.x * sumx[ii];
            float final_y = spin_constant * mxBrms.y * sumy[ii];
            float final_z = spin_constant * mxBrms.z * sumz[ii];

            // Second Summatory
            new_term_x[ii] += final_x;
            new_term_y[ii] += final_y;
            new_term_z[ii] += final_z;

//printf("new_term_x: %f\n",  new_term_x[ii]);
// printf("new_term_y: %f\n",  new_term_y[ii]);
//printf("new_term_z: %f\n",  new_term_z[ii]);
          }
}
