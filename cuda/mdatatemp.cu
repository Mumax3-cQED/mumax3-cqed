// MODIFIED INMA
#include "amul.h"
#include "constants.h"
#include "stencil.h"
#include <stdint.h>

extern "C" __global__ void
mdatatemp(float* __restrict__ new_term_x, float* __restrict__ new_term_y, float* __restrict__ new_term_z,
          float* __restrict__ dst_sin_x, float* __restrict__ dst_sin_y, float* __restrict__ dst_sin_z,
          float* __restrict__ dst_cos_x, float* __restrict__ dst_cos_y, float* __restrict__ dst_cos_z,
          float* __restrict__ sum_x, float* __restrict__ sum_y, float* __restrict__ sum_z,
          float* __restrict__ lx, float lx_mul,
          float* __restrict__ ly, float ly_mul,
          float* __restrict__ lz, float lz_mul,
          // float* __restrict__ alpha_, float alpha_mul,
          float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
          float delta_time, float ctimeWc,
          float brms_x, float brms_y, float brms_z,
          int N) {

          // int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
          //
          // if (i < N) {

          int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

          if (i < N) {

          // First Summatory
          // dst_sin[i] += amul(mx, sin(ctimeWc), i) + amul(my, sin(ctimeWc), i) + amul(mz, sin(ctimeWc), i);
          // dst_cos[i] += amul(mx, cos(ctimeWc), i) + amul(my, cos(ctimeWc), i) + amul(mz, cos(ctimeWc), i);

          // float result_sin = 0.0;
          // float result_cos = 0.0;

          // for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
          //     ii < N;
          //     ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {

                      dst_sin_x[i] += amul(mx, sin(ctimeWc), i);
                      dst_sin_y[i] += amul(my, sin(ctimeWc), i);
                      dst_sin_z[i] += amul(mz, sin(ctimeWc), i);

                      dst_cos_x[i] += amul(mx, cos(ctimeWc), i);
                      dst_cos_y[i] += amul(my, cos(ctimeWc), i);
                      dst_cos_z[i] += amul(mz, cos(ctimeWc), i);

                    //   float result_sin = dst_sin_x[i] + dst_sin_y[i] + dst_sin_z[i];
                    // float  result_cos = dst_cos_x[i] + dst_cos_y[i] + dst_cos_z[i];

                      // dst_sinx[idcell] += amul(mx, sin(ctimeWc), idcell);
                      // dst_siny[idcell] += amul(my, sin(ctimeWc), idcell);
                      // dst_sinz[idcell] += amul(mz, sin(ctimeWc), idcell);
                      //
                      // dst_cosx[idcell] += amul(mx, cos(ctimeWc), idcell);
                      // dst_cosy[idcell] += amul(my, cos(ctimeWc), idcell);
                      // dst_cosz[idcell] += amul(mz, cos(ctimeWc), idcell);

              //     }
              // }
          // }

          // Second summatory
          // float val_x = brms_x * delta_time * (amul(dst_sin, cos(ctimeWc), i) - amul(dst_cos, sin(ctimeWc), i));
          // float val_y = brms_y * delta_time * (amul(dst_sin, cos(ctimeWc), i) - amul(dst_cos, sin(ctimeWc), i));
          // float val_z = brms_z * delta_time * (amul(dst_sin, cos(ctimeWc), i) - amul(dst_cos, sin(ctimeWc), i));
          //
          // sum[i] += (val_x + val_y + val_z);

          // for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
          //     ii < Nx*Ny;
          //     ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {
          //
          //   sumx[ii] += brms_x * delta_time * (amul(dst_sinx, cos(ctimeWc), ii) - amul(dst_cosx, sin(ctimeWc), ii));
          //   sumy[ii] += brms_y * delta_time * (amul(dst_siny, cos(ctimeWc), ii) - amul(dst_cosy, sin(ctimeWc), ii));
          //   sumz[ii] += brms_z * delta_time * (amul(dst_sinz, cos(ctimeWc), ii) - amul(dst_cosz, sin(ctimeWc), ii));
          // }



          // float result_sum = 0.0;

        float result_sum = 0.0;
          for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
              ii < N;
              ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {



                sum_x[ii] += brms_x * delta_time * (dst_sin_x[ii] * cos(ctimeWc) - (dst_cos_x[ii] * sin(ctimeWc)));
                sum_y[ii] += brms_y * delta_time * (dst_sin_y[ii] * cos(ctimeWc) - (dst_cos_y[ii] * sin(ctimeWc)));
                sum_z[ii] += brms_z * delta_time * (dst_sin_z[ii] * cos(ctimeWc) - (dst_cos_z[ii] * sin(ctimeWc)));

                result_sum += sum_x[ii] + sum_y[ii] + sum_z[ii];
                      // sumx[idcell] += brms_x * delta_time * (amul(dst_sinx, cos(ctimeWc), idcell) - amul(dst_cosx, sin(ctimeWc), idcell));
                      // sumy[idcell] += brms_y * delta_time * (amul(dst_siny, cos(ctimeWc), idcell) - amul(dst_cosy, sin(ctimeWc), idcell));
                      // sumz[idcell] += brms_z * delta_time * (amul(dst_sinz, cos(ctimeWc), idcell) - amul(dst_cosz, sin(ctimeWc), idcell));
          }

          // float result_sum = sum_x[i] + sum_y[i] + sum_z[i];

          // float alpha = amul(alpha_, alpha_mul, i);
          // float gilb = 1;//1.0f / (1.0f + alpha * alpha);

          float3 m = {mx[i], my[i], mz[i]};
          float3 brms = {brms_x, brms_y, brms_z};

           // float3 p = 1*vmul(lx, ly, lz, lx_mul, ly_mul, lz_mul, i));
           // float3 pxm = cross(p, m);
           // float3 mxpxm = cross(m, pxm);

          float3 mxBrms = cross(m, brms); // m x Brms

          float spin_constant = 2 / HBAR; // debemos dividir entre gamma0 nuestro nuevo termino? parece que si

          // float result =  sum_x[i] +  sum_y[i] +  sum_z[i]
          // float3 final = result * spin_constant * mxBrms;

          float final_x = result_sum * spin_constant * mxBrms.x;
          float final_y = result_sum * spin_constant * mxBrms.y;
          float final_z = result_sum * spin_constant * mxBrms.z;

          // float3 norm = normalized(make_float3(final_x, final_y, final_z));
          // Final value
          new_term_x[i] = final_x;
          new_term_y[i] = final_y;
          new_term_z[i] = final_z;
       }
}
