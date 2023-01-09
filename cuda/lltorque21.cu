#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "constants.h"
#include "stencil.h"

// Landau-Lifshitz torque.
extern "C" __global__ void
lltorque21(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__ dst_sin_x, float* __restrict__ dst_sin_y, float* __restrict__ dst_sin_z,
          float* __restrict__ dst_cos_x, float* __restrict__ dst_cos_y, float* __restrict__ dst_cos_z,
          float* __restrict__ wc, float wc_mul,
          float* __restrict__ msat, float msat_mul,
          float* __restrict__ brms_x, float brmsx_mul,
          float* __restrict__ brms_y, float brmsy_mul,
          float* __restrict__ brms_z, float brmsz_mul,
          float delta_time, float ctime, float vol,
          float* __restrict__  alpha_, float alpha_mul, int Nx, int Ny, int Nz, uint8_t PBC) {

       int ix = blockIdx.x * blockDim.x + threadIdx.x;
       int iy = blockIdx.y * blockDim.y + threadIdx.y;
       int iz = blockIdx.z * blockDim.z + threadIdx.z;

       if (ix >= Nx || iy >= Ny || iz >= Nz) {
           return;
       }

        int i = idx(ix, iy, iz);

    // int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // if (i < N) {

    float wc_val = amul(wc, wc_mul, i);
    float msat_val = amul(msat, msat_mul, i);

    float brmsx = amul(brms_x, brmsx_mul, i);
    float brmsy = amul(brms_y, brmsy_mul, i);
    float brmsz = amul(brms_z, brmsz_mul, i);

    float dt = delta_time/GAMMA0;

    ////////// START IMPLEMENTACION 1

    // First summatory
    float cell_sum = 0.0;
    for (int c = 0; c < Nz; c++) {
      for (int b = 0; b < Ny; b++) {
        for (int a = 0; a < Nx; a++) {

          int ii = idx(a, b, c);

          float sum_resx = amul(mx, brmsx, ii);
          float sum_resy = amul(my, brmsy, ii);
          float sum_resz = amul(mz, brmsz, ii);

          cell_sum += (sum_resx + sum_resy + sum_resz);
          //dst_sin_y[ii] += (sum_resx + sum_resy + sum_resz);
        }
      }
    }

    // float sum_resx = amul(mx, brmsx, 0);
    // float sum_resy = amul(my, brmsy, i);
    // float sum_resz = amul(mz, brmsz, i);

    // dst_sin_y[i] += (sum_resx + sum_resy + sum_resz);

    float3 brms = {brmsx, brmsy, brmsz};
    float prefactor = (2 / HBAR) * vol * msat_val;

    dst_sin_x[i] += sin(wc_val * ctime) * cell_sum * dt;
    dst_cos_x[i] += cos(wc_val * ctime) * cell_sum * dt;

    float gamma =  prefactor * ((cos(wc_val * ctime) * dst_sin_x[i]) - (sin(wc_val * ctime) * dst_cos_x[i]));

    float3 bext = brms * gamma;

    ////////// END IMPLEMENTACION 1


    ////////// START IMPLEMENTACION 2
    // Second summatory
    // dst_sin_x[i] += sin(wc_val * ctime) * sum_resx * dt;
    // dst_sin_y[i] += sin(wc_val * ctime) * sum_resy * dt;
    // dst_sin_z[i] += sin(wc_val * ctime) * sum_resz * dt;
    //
    // dst_cos_x[i] += cos(wc_val * ctime) * sum_resx * dt;
    // dst_cos_y[i] += cos(wc_val * ctime) * sum_resy * dt;
    // dst_cos_z[i] += cos(wc_val * ctime) * sum_resz * dt;

    // // First summatory
    // dst_sin_x[i] += amul(mx, sin(ctime * wc_val), i);
    // dst_sin_y[i] += amul(my, sin(ctime * wc_val), i);
    // dst_sin_z[i] += amul(mz, sin(ctime * wc_val), i);
    //
    // dst_cos_x[i] += amul(mx, cos(ctime * wc_val), i);
    // dst_cos_y[i] += amul(my, cos(ctime * wc_val), i);
    // dst_cos_z[i] += amul(mz, cos(ctime * wc_val), i);
    //
    // __syncthreads();
    //
    // // Second summatory
    // float result_sum = 0.0;
    //
    // for (int c = 0; c < Nz; c++) {
    //   for (int b = 0; b < Ny; b++) {
    //     for (int a = 0; a < Nx; a++) {
    //
    //         int ii = idx(a, b, c);
    //
    //         // First summatory
    //         dst_sin_x[ii] += amul(mx, sin(ctime * wc_val), ii);
    //         dst_sin_y[ii] += amul(my, sin(ctime * wc_val), ii);
    //         dst_sin_z[ii] += amul(mz, sin(ctime * wc_val), ii);
    //
    //         dst_cos_x[ii] += amul(mx, cos(ctime * wc_val), ii);
    //         dst_cos_y[ii] += amul(my, cos(ctime * wc_val), ii);
    //         dst_cos_z[ii] += amul(mz, cos(ctime * wc_val), ii);
    //
    //         // __syncthreads();
    //
    //         float sum_x = brmsx * dt * ((dst_sin_x[ii] * cos(ctime * wc_val) - dst_cos_x[ii] * sin(ctime * wc_val)));
    //         float sum_y = brmsy * dt * ((dst_sin_y[ii] * cos(ctime * wc_val) - dst_cos_y[ii] * sin(ctime * wc_val)));
    //         float sum_z = brmsz * dt * ((dst_sin_z[ii] * cos(ctime * wc_val) - dst_cos_z[ii] * sin(ctime * wc_val)));
    //
    //         result_sum += (sum_x + sum_y + sum_z);
    //     }
    //   }
    // }
    //
    //__syncthreads();
    //
    //float3 brms = {brmsx, brmsy, brmsz};
    //float prefactor = (2 / HBAR) * vol * msat_val;
    //
    //float3 torque = prefactor * result_sum * brms;

    ////////// END IMPLEMENTACION 2

    //printf("torque.x: %.8f\n", torque.x);
    //printf(torque.y: "%.8f\n", torque.y);
    //printf(torque.z: "%.8f\n", torque.z);

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};
        float alpha = amul(alpha_, alpha_mul, i);

        float3 mxH =  cross(m, H - bext);
        float gilb = -1.0f / (1.0f + alpha * alpha);

        float3 torque = gilb  * (mxH + alpha * cross(m, mxH));

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    // }
}
