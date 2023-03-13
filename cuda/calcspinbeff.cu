// MODIFIED INMA
#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "constants.h"
#include "stencil.h"

// Bcustom amount calculation.
extern "C" __global__ void
calcspinbeff(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__ sn, float* __restrict__ cn,
          float* __restrict__ cell_sum,
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

    // int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // if (i < N) {

    float wc_val = amul(wc, wc_mul, i);
    float msat_val = amul(msat, msat_mul, i);

    float brmsx = amul(brms_x, brmsx_mul, i);
    float brmsy = amul(brms_y, brmsy_mul, i);
    float brmsz = amul(brms_z, brmsz_mul, i);

    ////////// START IMPLEMENTACION 1

    // First summatory
    // float cell_sum2 = 0.0f;
    // for (int c = 0; c < Nz; c++) {
    //   for (int b = 0; b < Ny; b++) {
    //      for (int a =0; a < Nx; a++) {
    float tmpSum = 0.0;
    for (int c = blockIdx.z * blockDim.z + threadIdx.z; c < Nz; c += blockDim.z * gridDim.z) {
      for (int b = blockIdx.y * blockDim.y + threadIdx.y; b < Ny; b += blockDim.y * gridDim.y) {
         for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < Nx; a += blockDim.x * gridDim.x) {

           int ii = idx(a, b, c);

           float sum_resx = mx[ii] * brmsx;
           float sum_resy = my[ii] * brmsy;
           float sum_resz = mz[ii] * brmsz;

          tmpSum  += (sum_resx + sum_resy + sum_resz);
           // sum2 += (sum_resx + sum_resy + sum_resz);

        }
      }
    }

    // int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    // int COL = blockIdx.x*blockDim.x+threadIdx.x;

    // float tmpSum = 0;
    // int N = Nx*Ny*Nz;
    // // if (ROW < N && COL < N) {
    //     // each thread computes one element of the block sub-matrix
    //     for (int ii = 0; ii < N; i++) {
    //
    //       float sum_resx = mx[ii] * brmsx;
    //       float sum_resy = my[ii] * brmsy;
    //       float sum_resz = mz[ii] * brmsz;
    //         tmpSum += (sum_resx + sum_resy + sum_resz);//A[ROW * N + i] * B;
    //     }
    // }
    // cell_sum[ROW * N + COL] = tmpSum;
    cell_sum[i] = tmpSum;

    float PREFACTOR = (2 / HBAR) * vol * msat_val;
    float dt = delta_time/GAMMA0;

    // Second summatory
    sn[i] += sin(wc_val * ctime) * cell_sum[i] * dt;
    cn[i] += cos(wc_val * ctime) * cell_sum[i] * dt;

    // atomicAdd(&sn[i], sin(wc_val * ctime) * cell_sum2 * dt);
    // atomicAdd(&cn[i], cos(wc_val * ctime) * cell_sum2 * dt);

    float gamma = PREFACTOR * ((cos(wc_val * ctime) * sn[i]) - (sin(wc_val * ctime) * cn[i]));

    float3 brms = {brmsx, brmsy, brmsz};
    float3 bext = brms * gamma;

    // Beff - Bcustom
    tx[i] -= bext.x;
    ty[i] -= bext.y;
    tz[i] -= bext.z;
}
