// MODIFIED INMA
#include "stencil.h"
#include "amul.h"

extern "C" __global__ void
mdatatemp(float* __restrict__ dst_sinx, float* __restrict__ dst_siny, float* __restrict__ dst_sinz,
          float* __restrict__ dst_cosx, float* __restrict__ dst_cosy, float* __restrict__ dst_cosz,
          float* __restrict__ delta_time, float* __restrict__ brms_x, float* __restrict__ brms_y, float* __restrict__ brms_z,
          float* __restrict__ current_mx, float* __restrict__ current_my, float* __restrict__ current_mz,
          float ctimeWc, float h_delta, float brmsx, float brmsy, float brmsz, int N) {

          int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

          if (i < N) {

              // First Summatory
              dst_sinx[i] += amul(current_mx, sin(ctimeWc), i);
              dst_siny[i] += amul(current_my, sin(ctimeWc), i);
              dst_sinz[i] += amul(current_mz, sin(ctimeWc), i);

              dst_cosx[i] += amul(current_mx, cos(ctimeWc), i);
              dst_cosy[i] += amul(current_my, cos(ctimeWc), i);
              dst_cosz[i] += amul(current_mz, cos(ctimeWc), i);

              delta_time[i] = h_delta;
              brms_x[i] = brmsx;
              brms_y[i] = brmsy;
              brms_z[i] = brmsz;
          }
}
