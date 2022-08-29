// MODIFIED INMA

#include "amul.h"
#include "constants.h"

extern "C" __global__ void
term2time(float* __restrict__  new_term_x, float* __restrict__  new_term_y, float* __restrict__  new_term_z,
          float* __restrict__ rk_sin_mx, float* __restrict__ rk_sin_my, float* __restrict__ rk_sin_mz,
          float* __restrict__ rk_cos_mx, float* __restrict__ rk_cos_my, float* __restrict__ rk_cos_mz,
          float* __restrict__ delta_time, float* __restrict__ brms_x, float* __restrict__ brms_y, float* __restrict__ brms_z,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__ sumx, float* __restrict__ sumy, float* __restrict__ sumz,
          float ctimeWc, int N) {

          int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

          if (i < N) {

            float3 m = {mx[i], my[i], mz[i]};

            float3 brms = {brms_x[i], brms_y[i], brms_z[i]};

            sumx[i] += amul(brms_x, amul(delta_time, amul(rk_sin_mx, cos(ctimeWc), i) - amul(rk_cos_mx, sin(ctimeWc), i), i), i);
            sumy[i] += amul(brms_y, amul(delta_time, amul(rk_sin_my, cos(ctimeWc), i) - amul(rk_cos_my, sin(ctimeWc), i), i), i);
            sumz[i] += amul(brms_z, amul(delta_time, amul(rk_sin_mz, cos(ctimeWc), i) - amul(rk_cos_mz, sin(ctimeWc), i), i), i);

            // Creating new time-dependant term
            float3 mxBrms = cross(m, brms); // m x Brms
            float spin_constant = 2 / HBAR; // debemos dividir entre gamma0 nuestro nuevo termino? parece que si

            float final_x = spin_constant * amul(sumx, mxBrms.x, i);
            float final_y = spin_constant * amul(sumy, mxBrms.y, i);
            float final_z = spin_constant * amul(sumz, mxBrms.z, i);

            // Second Summatory
            new_term_x[i] += final_x;
            new_term_y[i] += final_y;
            new_term_z[i] += final_z;

//printf("new_term_x: %f\n",  new_term_x[i]);
// printf("new_term_y: %f\n",  new_term_y[i]);
//printf("new_term_z: %f\n",  new_term_z[i]);
          }
}
