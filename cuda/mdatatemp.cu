// MODIFIED INMA
#include "amul.h"
#include "constants.h"
#include "stencil.h"
#include <stdint.h>

extern "C" __global__ void
mdatatemp(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
          float* __restrict__ dst_sin_x,
          float* __restrict__ dst_sin_y,
          float* __restrict__ dst_sin_z,
          float* __restrict__ dst_cos_x,
          float* __restrict__ dst_cos_y,
          float* __restrict__ dst_cos_z,
          float* __restrict__ sum_x,
          float* __restrict__ sum_y,
          float* __restrict__ sum_z,
          float* __restrict__ wc, float wc_mul,
          float* __restrict__ brms_x, float brmsx_mul,
          float* __restrict__ brms_y, float brmsy_mul,
          float* __restrict__ brms_z, float brmsz_mul,
          float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
          float delta_time, float ctime,
          int Nx, int Ny, int Nz, uint8_t PBC) {

          // int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
          //
          // if (i < N) {
          int ix = blockIdx.x * blockDim.x + threadIdx.x;
          int iy = blockIdx.y * blockDim.y + threadIdx.y;
          int iz = blockIdx.z * blockDim.z + threadIdx.z;

          if (ix >= Nx || iy >= Ny || iz >= Nz) {
              return;
          }

           int i = idx(ix, iy, iz);
          // int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
          //
          // if (i < N) {

              float wc_val = amul(wc, wc_mul, i);

              float brmsx = amul(brms_x, brmsx_mul, i);
              float brmsy = amul(brms_y, brmsy_mul, i);
              float brmsz = amul(brms_z, brmsz_mul, i);

          // First Summatory
          // dst_sin[i] += amul(mx, sin(ctimeWc), i) + amul(my, sin(ctimeWc), i) + amul(mz, sin(ctimeWc), i);
          // dst_cos[i] += amul(mx, cos(ctimeWc), i) + amul(my, cos(ctimeWc), i) + amul(mz, cos(ctimeWc), i);

          // float result_sin = 0.0;
          // float result_cos = 0.0;


          // for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
          //     ii < N;
          //     ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {

                      dst_sin_x[i] += amul(mx, sin(ctime * wc_val), i);
                      dst_sin_y[i] += amul(my, sin(ctime * wc_val), i);
                      dst_sin_z[i] += amul(mz, sin(ctime * wc_val), i);

                      dst_cos_x[i] += amul(mx, cos(ctime * wc_val), i);
                      dst_cos_y[i] += amul(my, cos(ctime * wc_val), i);
                      dst_cos_z[i] += amul(mz, cos(ctime * wc_val), i);

                      // __syncthreads();

                      // float result_sin = dst_sin_x[i] + dst_sin_y[i] + dst_sin_z[i];
                      // float result_cos = dst_cos_x[i] + dst_cos_y[i] + dst_cos_z[i];

                      // dst_sinx[idcell] += amul(mx, sin(ctimeWc), idcell);
                      // dst_siny[idcell] += amul(my, sin(ctimeWc), idcell);
                      // dst_sinz[idcell] += amul(mz, sin(ctimeWc), idcell);
                      //
                      // dst_cosx[idcell] += amul(mx, cos(ctimeWc), idcell);
                      // dst_cosy[idcell] += amul(my, cos(ctimeWc), idcell);
                      // dst_cosz[idcell] += amul(mz, cos(ctimeWc), idcell);

           //         }
           //     }
           // }

          // Second summatory
          // float val_x = brms_x * delta_time * (amul(dst_sin_x, cos(ctime * wc_val), i) - amul(dst_cos_x, sin(ctime * wc_val), i));
          // float val_y = brms_y * delta_time * (amul(dst_sin_y, cos(ctime * wc_val), i) - amul(dst_cos_y, sin(ctime * wc_val), i));
          // float val_z = brms_z * delta_time * (amul(dst_sin_z, cos(ctime * wc_val), i) - amul(dst_cos_z, sin(ctime * wc_val), i));
          //
          // sum_x[i] += (val_x + val_y + val_z);

           float result_sum = 0.0;
         //  // for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
         //  //     ii <N;
         //  //     ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {
          for (int c = 0; c < Nz; c++) {
            for (int b = 0; b < Ny; b++) {
               for (int a =0; a < Nx; a++) {

                 int ii = idx(a, b, c);
                sum_x[ii] = brmsx * delta_time/GAMMA0 * (dst_sin_x[ii] * cos(ctime * wc_val) - dst_cos_x[ii] * sin(ctime * wc_val));
                sum_y[ii] = brmsy * delta_time/GAMMA0 * (dst_sin_y[ii] * cos(ctime * wc_val) - dst_cos_y[ii] * sin(ctime * wc_val));
                sum_z[ii] = brmsz * delta_time/GAMMA0 * (dst_sin_z[ii] * cos(ctime * wc_val) - dst_cos_z[ii] * sin(ctime * wc_val));

                result_sum += (sum_x[ii] + sum_y[ii] + sum_z[ii]);
             }
           }
         }
          // }

          // __syncthreads();

          // sum_x[i] += brmsx * delta_time * (dst_sin_x[i] * cos(ctime * wc_val) - (dst_cos_x[i] * sin(ctime * wc_val)));
          // sum_y[i] += brmsy * delta_time * (dst_sin_y[i] * cos(ctime * wc_val) - (dst_cos_y[i] * sin(ctime * wc_val)));
          // sum_z[i] += brmsz * delta_time * (dst_sin_z[i] * cos(ctime * wc_val) - (dst_cos_z[i] * sin(ctime * wc_val)));

          // float result_sum = 0.0;

         // float result_sum = 0.0;
         //  for (int ii = (blockIdx.y * blockDim.y + threadIdx.y) * blockDim.x * gridDim.x + (blockIdx.x * blockDim.x + threadIdx.x);
         //      ii < N;
         //      ii += blockDim.y * gridDim.y * blockDim.x * gridDim.x) {




                // float result_sum = sum_x[i] + sum_y[i] + sum_z[i];

                      // sumx[idcell] += brms_x * delta_time * (amul(dst_sinx, cos(ctimeWc), idcell) - amul(dst_cosx, sin(ctimeWc), idcell));
                      // sumy[idcell] += brms_y * delta_time * (amul(dst_siny, cos(ctimeWc), idcell) - amul(dst_cosy, sin(ctimeWc), idcell));
                      // sumz[idcell] += brms_z * delta_time * (amul(dst_sinz, cos(ctimeWc), idcell) - amul(dst_cosz, sin(ctimeWc), idcell));
          // }


          // float result_sum = sum_x[i] + sum_y[i] + sum_z[i];

             // float alpha = amul(alpha_, alpha_mul, i);
            // float gilb =  1.0f / (1.0f + alpha * alpha);

          float3 m = {mx[i], my[i], mz[i]};
          float3 brms = {brmsx, brmsy, brmsz};

          // float3 p = 1*vmul(lx, ly, lz, lx_mul, ly_mul, lz_mul, i);
          //
           // float3 mxpxm = cross(m, pxm);
 // float3 p = normalized(vmul(lx, ly, lz, lx_mul, ly_mul, lz_mul, i));
 // float3 pxm = cross(p, m);
          float3 mxBrms = cross(m, brms); // m x Brms

          float spin_constant = 2 / HBAR; // debemos dividir entre gamma0 nuestro nuevo termino? parece que si

          // float result =  sum_x[i] +  sum_y[i] +  sum_z[i]
          // float3 final = result * spin_constant * mxBrms;

           float3 torque = spin_constant * sum_x[i] * mxBrms;

           // float final_x = gilb * result_sum * spin_constant * mxBrms.x;
           // float final_y = gilb * result_sum * spin_constant * mxBrms.y;
           // float final_z = gilb * result_sum * spin_constant * mxBrms.z;

          //   float3 norm = normalized(torque);
          //   printf("%.8f\n", norm.x);
          // Final value
          tx[i] -= torque.x;
          ty[i] -= torque.y;
          tz[i] -= torque.z;
       // }
}
