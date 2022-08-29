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
// lltorque2time(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
//           float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
//           float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
//           float* __restrict__  alpha_, float alpha_mul,
//           float* __restrict__ rk_sin_mx, float* __restrict__ rk_sin_my, float* __restrict__ rk_sin_mz,
//           float* __restrict__ rk_cos_mx, float* __restrict__ rk_cos_my, float* __restrict__ rk_cos_mz,
//           float* __restrict__ delta_time, float* __restrict__ brms_x, float* __restrict__ brms_y, float* __restrict__ brms_z,
//           float* __restrict__ sumx, float* __restrict__ sumy, float* __restrict__ sumz,
//           float ctimeWc, int dt_gt, int Nx, int Ny, int Nz) {

// lltorque2time(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
//           float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
//           float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
//           float* __restrict__  alpha_, float alpha_mul,
//           float* __restrict__ new_term_x, float* __restrict__ new_term_y, float* __restrict__ new_term_z,
//           float ctimeWc, int dt_gt, int N) {

lltorque2time(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
              float* __restrict__ new_term_x, float* __restrict__ new_term_y, float* __restrict__ new_term_z, int N) {

     int ii =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

     if (ii < N) {
        // int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);
        // int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
        // int iz = ((blockIdx.z * blockDim.z) + threadIdx.z);
        //
        // if (ix >= Nx || iy >= Ny || iz >= Nz) {
        //     return;
        // }
        //
        // int ii = idx(ix, iy, iz);

        // float3 m = {mx[ii], my[ii], mz[ii]};
        // float3 H = {hx[ii], hy[ii], hz[ii]};
        // float alpha = amul(alpha_, alpha_mul, ii);
        //
        // float3 mxH = cross(m, H);
        // float gilb = -1.0f / (1.0f + alpha * alpha);
        //
        //   float3 torque = make_float3(0.0, 0.0, 0.0);

      // printf("Hello thread %d, f=%d\n", threadIdx.x, dt_gt);

         // if (dt_gt == 1) {

            // Summatory for all cells

              // for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
              //       ii < N; ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {

        //       int jx = ((blockIdx.x * blockDim.x) + threadIdx.x);
        //       int jy = ((blockIdx.y * blockDim.y) + threadIdx.y);
        //       int jz = ((blockIdx.z * blockDim.z) + threadIdx.z);
        //
        //       if (jx >= Nx || jy >= Ny || jz >= Nz) {
        //           return;
        //       }
        //
        //       int jj = idx(jx, jy, jz);
        //      //
        //      float3 mm =  make_float3(mx[jj], my[jj], mz[jj]);
        //
        //      float3 brms =  make_float3(brms_x[jj], brms_y[jj], brms_z[jj]);
        //      //
        //      sumx[jj] +=  (brms_x[jj] * delta_time[jj]) * (cos(ctimeWc) * rk_sin_mx[jj] - sin(ctimeWc) * rk_cos_mx[jj]);// funciona
        //      sumy[jj] +=  (brms_y[jj] * delta_time[jj]) * (cos(ctimeWc) * rk_sin_my[jj] - sin(ctimeWc) * rk_cos_my[jj]);
        //      sumz[jj] +=  (brms_z[jj] * delta_time[jj]) * (cos(ctimeWc) * rk_sin_mz[jj] - sin(ctimeWc) * rk_cos_mz[jj]);
        //
        //     // Adding new time-dependant term to equation
        //     float3 mxBrms = cross(mm, brms); // m x Brms
        //     //
        //      float spin_constant = 2 / HBAR; // debemos dividir entre gamma0 nuestro nuevo termino? parece que si
        //     sumx[jj] = spin_constant * mxBrms.x * sumx[jj];
        //     sumy[jj] = spin_constant * mxBrms.y * sumy[jj];
        //     sumz[jj] = spin_constant * mxBrms.z * sumz[jj];
        //     //
        //     float3 new_term = make_float3(sumx[jj], sumy[jj], sumz[jj]);
        //   //  printf("rk_sin_m[i]: %f\n", rk_sin_m.x);
        //     // printf("rk_sin_m[i]: %f\n", rk_cos_m.x);
        //     // printf("Hello thread %d, f=%f\n", threadIdx.x, new_term);
        //     // float3 new_term = {new_term_x[ii], new_term_y[ii], new_term_x[ii]};
        //
        //    torque = gilb * (mxH + alpha * cross(m, mxH)) - new_term;
        //
        //    // sumx[jj] = 0.0;
        //    // sumy[jj] = 0.0;
        //    // sumz[jj] = 0.0;

        // float3 new_term = make_float3(new_term_x[ii], new_term_y[ii], new_term_z[ii]);
        //           torque = gilb * (mxH + alpha * cross(m, mxH)) - new_term;  // LLG equation with full new time-dependant term to plug in equation

          // Apply new term to torque
          tx[ii] -= new_term_x[ii];
          ty[ii] -= new_term_y[ii];
          tz[ii] -= new_term_z[ii];

         // }
        //  else {
        //     // torque = gilb * (mxH + alpha * cross(m, mxH));
        //
        //     tx[ii] = torque.x;
        //     ty[ii] = torque.y;
        //     tz[ii] = torque.z;
         // }


        }
}
