// MODIFIED INMA

#include "amul.h"
#include "constants.h"
#include "stencil.h"
#include <stdio.h>

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
          float* __restrict__ rk_cos_mx, float* __restrict__ rk_cos_my, float* __restrict__ rk_cos_mz,
          float* __restrict__ delta_time, float* __restrict__ brms_x, float* __restrict__ brms_y, float* brms_z,
          float* __restrict__ sumx, float* __restrict__ sumy, float* __restrict__ sumz,
          float ctimeWc, int dt_gt, int Nx, int Ny, int Nz, int N) {

    // int ii =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // //
    // if (ii < N) {
        int ix = ((blockIdx.x * blockDim.x) + threadIdx.x);
        int iy = ((blockIdx.y * blockDim.y) + threadIdx.y);
        int iz = ((blockIdx.z * blockDim.z) + threadIdx.z);

        if (ix >= Nx || iy >= Ny || iz >= Nz) {
            return;
        }

        int ii = idx(ix, iy, iz);

        float3 m = {mx[ii], my[ii], mz[ii]};
        float3 H = {hx[ii], hy[ii], hz[ii]};
        float alpha = amul(alpha_, alpha_mul, ii);

        float3 mxH = cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);

        float3 torque = {0.0, 0.0, 0.0};

      // printf("Hello thread %d, f=%d\n", threadIdx.x, dt_gt);

         if (dt_gt == 1) {

            // Summatory for all cells

              // for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
              //       ii < N; ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {
              float3 sum_final = {0.0, 0.0, 0.0};

              int jx = ((blockIdx.x * blockDim.x) + threadIdx.x);
              int jy = ((blockIdx.y * blockDim.y) + threadIdx.y);
              int jz = ((blockIdx.z * blockDim.z) + threadIdx.z);

              if (jx >= Nx || jy >= Ny || jz >= Nz) {
                  return;
              }

              int jj = idx(jx, jy, jz);

             float3 rk_sin_m = {rk_sin_mx[jj], rk_sin_my[jj], rk_sin_mz[jj]};
             float3 rk_cos_m = {rk_cos_mx[jj], rk_cos_my[jj], rk_cos_mz[jj]};

             float3 mm = {mx[jj], my[jj], mz[jj]};

             float3 brms = {brms_x[jj], brms_y[jj], brms_z[jj]};
            //
            // sumx[ii] += (brms_x[ii] * delta_time[ii]) * (rk_sin_mx[ii] - rk_cos_mx[ii]);
            // sumy[ii] += (brms_y[ii] * delta_time[ii]) * (rk_sin_my[ii] - rk_cos_my[ii]);
            // sumy[ii] += (brms_z[ii] * delta_time[ii]) * (rk_sin_mz[ii] - rk_cos_mz[ii]);

            // float op1_left = (brms_x[ii] * delta_time[ii] * cos(ctimeWc));
            // float op2_left = (brms_y[ii] * delta_time[ii] * cos(ctimeWc));
            // float op3_left = (brms_z[ii] * delta_time[ii] * cos(ctimeWc));
            //
            // float3 left_side = vmul(rk_sin_mx, rk_sin_my, rk_sin_mz, op1_left, op2_left, op3_left, ii);
            //
            // float op_right = sin(ctimeWc);
            // float3 right_side = vmul(rk_cos_mx, rk_cos_my, rk_cos_mz, op_right, op_right, op_right, ii);
            // sumx[ii] += operation_final_x;
            // sumy[ii] += operation_final_y;
            // sumz[ii] += operation_final_z;
  //   printf("rk_sin_mx[ii] %LF\n", rk_sin_mx[ii]);
  // printf("rk_cos_mx[ii] %LF\n", rk_cos_mx[ii]);
  //  printf("substract[ii] %LF\n", rk_sin_mx[ii]- rk_cos_mx[ii]);
 // printf("delta_time[ii] %f\n", delta_time[ii]);
 // printf("rk_sin_mx[ii] %f\n", rk_sin_mx[ii]);
 //  printf("rk_cos_mz[ii] %f\n", rk_cos_mz[ii]);

// printf("ctime[ii] %f\n", ctime[ii]);

            // printf("ix %f\n", ix);
            // printf("iy %f\n", iy);
            // printf("iz %f\n", iz);
            //
             // printf("sumx %lf\n", sumx[ii]);
             // printf("sumy %lf\n", sumy[ii]);
             // printf("sumz %lf\n", sumz[ii]);

            //__syncthreads();
             //printf("sumx[ii] %f\n", sumx[ii]);
            // printf("sumy[ii] %f\n", sumy[ii]);
            // printf("sumz[ii] %f\n", sumz[ii]);
            // float3 sum_final = {sumx[ii], sumy[ii], sumz[ii]}; //  left_side - right_side;
             sum_final +=  (brms * delta_time[jj]) * (cos(ctimeWc) * rk_sin_m - sin(ctimeWc) * rk_cos_m);// funciona
            // *
             // }
 //             sumx[ii] += sum_final_temp.x;
 // sumy[ii] += sum_final_temp.y;
 //  sumz[ii] += sum_final_temp.z;
 //    float3 sum_final = {sumx[ii], sumy[ii], sumz[ii]};
            // Adding new time-dependant term to equation
            float3 mxBrms = cross(mm, brms); // m x Brms

            float spin_constant = 2 / HBAR; // debemos dividir entre gamma0 nuestro nuevo termino? parece que si
            float3 new_term = spin_constant * mxBrms * sum_final;
          //  printf("rk_sin_m[i]: %f\n", rk_sin_m.x);
            // printf("rk_sin_m[i]: %f\n", rk_cos_m.x);
            // printf("Hello thread %d, f=%f\n", threadIdx.x, new_term);

           torque = gilb * (mxH + alpha * cross(m, mxH)) - new_term;
        } else {
           torque = gilb * (mxH + alpha * cross(m, mxH));
        }

        // float3 torque = gilb * (mxH + alpha * cross(m, mxH)) - new_term;  // LLG equation with full new time-dependant term to plug in equation

        tx[ii] = torque.x;
        ty[ii] = torque.y;
        tz[ii] = torque.z;
    // }
}
