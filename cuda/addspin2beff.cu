// MODIFIED INMA
#include "amul.h"
#include "constants.h"
#include "stencil.h"
#include <stdint.h>

extern "C" __global__ void
addspin2beff(float* __restrict__ tx, float* __restrict__ ty, float* __restrict__ tz,
             float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
             float* __restrict__ dst_sin_x, float* __restrict__ dst_sin_y, float* __restrict__ dst_sin_z,
             float* __restrict__ dst_cos_x, float* __restrict__ dst_cos_y, float* __restrict__ dst_cos_z,
             float* __restrict__ wc, float wc_mul,
             float* __restrict__ msat, float msat_mul,
             float* __restrict__ brms_x, float brmsx_mul,
             float* __restrict__ brms_y, float brmsy_mul,
             float* __restrict__ brms_z, float brmsz_mul,
             float delta_time, float ctime, float vol, int Nx, int Ny, int Nz, uint8_t PBC) {

          int ix = blockIdx.x * blockDim.x + threadIdx.x;
          int iy = blockIdx.y * blockDim.y + threadIdx.y;
          int iz = blockIdx.z * blockDim.z + threadIdx.z;

          if (ix >= Nx || iy >= Ny || iz >= Nz) {
              return;
          }

          int i = idx(ix, iy, iz);

          float wc_val = amul(wc, wc_mul, i);
          float msat_val = amul(msat, msat_mul, i);

          float brmsx = amul(brms_x, brmsx_mul, i);
          float brmsy = amul(brms_y, brmsy_mul, i);
          float brmsz = amul(brms_z, brmsz_mul, i);

          // First summatory
          dst_sin_x[i] += amul(mx, sin(ctime * wc_val), i);
          dst_sin_y[i] += amul(my, sin(ctime * wc_val), i);
          dst_sin_z[i] += amul(mz, sin(ctime * wc_val), i);

          dst_cos_x[i] += amul(mx, cos(ctime * wc_val), i);
          dst_cos_y[i] += amul(my, cos(ctime * wc_val), i);
          dst_cos_z[i] += amul(mz, cos(ctime * wc_val), i);

          __syncthreads();

          // Second summatory
          float result_sum = 0.0;

          for (int c = 0; c < Nz; c++) {
            for (int b = 0; b < Ny; b++) {
              for (int a = 0; a < Nx; a++) {

                  int ii = idx(a, b, c);

                  float sum_x = brmsx * (delta_time/GAMMA0) * ((dst_sin_x[ii] * cos(ctime * wc_val) - dst_cos_x[ii] * sin(ctime * wc_val)));
                  float sum_y = brmsy * (delta_time/GAMMA0) * ((dst_sin_y[ii] * cos(ctime * wc_val) - dst_cos_y[ii] * sin(ctime * wc_val)));
                  float sum_z = brmsz * (delta_time/GAMMA0) * ((dst_sin_z[ii] * cos(ctime * wc_val) - dst_cos_z[ii] * sin(ctime * wc_val)));

                  result_sum += (sum_x + sum_y + sum_z);
              }
            }
          }

          __syncthreads();

          float prefactor = (2 / HBAR) * vol * msat_val;
          float3 brms = {brmsx, brmsy, brmsz};

          float3 torque = prefactor * result_sum * brms;

          //printf("torque.x: %.8f\n", torque.x);
          //printf(torque.y: "%.8f\n", torque.y);
          //printf(torque.z: "%.8f\n", torque.z);

          // Final value
          tx[i] -= torque.x;
          ty[i] -= torque.y;
          tz[i] -= torque.z;
}
