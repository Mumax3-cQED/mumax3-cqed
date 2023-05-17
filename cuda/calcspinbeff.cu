// MODIFIED INMA
#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "constants.h"
#include "stencil.h"

#define idx3d(ix,iy,iz) ( ix + iy*Nx + iz*Nx*Ny )

// Note that warpReduce is a custom function that sums the input across threads in a warp using warp-synchronous programming
// This function uses the shuffle operation (__shfl_down_sync) to perform a pairwise reduction of the input value across threads in a warp
// The warpSize constant is used to determine the number of threads in a warp
// The function returns the final reduced sum to the caller
__inline__ __device__ float warpReduce(float val) {

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__inline__ __device__ float loopcells(float* mx, float* my, float* mz, float brmsx, float brmsy, float brmsz, int idx) {

  // float3 mm = {mx[idx], my[idx], mz[idx]};
  //     float3 bbrms = {brmsx[idx], brmsy[idx], brmsz[idx]};
  //     float sum_res = dot(mm, bbrms);

      //  sum_res = warpReduce(sum_res);
      // return sum_res;
    float sum_resx = amul(mx, brmsx, idx);
    float sum_resy = amul(my, brmsy, idx);
    float sum_resz = amul(mz, brmsz, idx);

    // Use warp-synchronous programming to sum results across threads
    sum_resx = warpReduce(sum_resx);
    sum_resy = warpReduce(sum_resy);
    sum_resz = warpReduce(sum_resz);

    return (sum_resx + sum_resy + sum_resz);
}

// Bcustom amount calculation.
extern "C" __global__ void
calcspinbeff(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__ sn, float* __restrict__ cn,
          float* __restrict__ wc, float wc_mul,
          float* __restrict__ nspins, float nspins_mul,
          float* __restrict__ brms_x, float brmsx_mul,
          float* __restrict__ brms_y, float brmsy_mul,
          float* __restrict__ brms_z, float brmsz_mul,
          float delta_time, float ctime, float gammaLL, int Nx, int Ny, int Nz,  uint8_t PBC) {
        //float delta_time, float ctime, float delta_vol, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
       return;
    }
    //
    int i = idx(ix, iy, iz);
    //int i = idx3d(ix, iy, iz);
    // int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    // if (i < N) {
    float wc_val = amul(wc, wc_mul, i);
    float nspins_val = amul(nspins, nspins_mul, i);

    float brmsx = amul(brms_x, brmsx_mul, i);
    float brmsy = amul(brms_y, brmsy_mul, i);
    float brmsz = amul(brms_z, brmsz_mul, i);

    // Classic loop
    // First summatory
    // float sum_cells = 0.0;
    // for (int c = blockIdx.z * blockDim.z + threadIdx.z; c < Nz; c += blockDim.z * gridDim.z) {
    //   for (int b = blockIdx.y * blockDim.y + threadIdx.y; b < Ny; b += blockDim.y * gridDim.y) {
    //      for (int a = blockIdx.x * blockDim.x + threadIdx.x; a < Nx; a += blockDim.x * gridDim.x) {
    //
    //        int ii = idx(a, b, c);
    //
    //        float sum_resx = mx[ii] * brmsx;
    //        float sum_resy = my[ii] * brmsy;
    //        float sum_resz = mz[ii] * brmsz;
    //
    //        sum_cells += (sum_resx + sum_resy + sum_resz);
    //     }
    //   }
    // }

            // float sum_resx = mx[i] * brmsx;
            // float sum_resy = my[i] * brmsy;
            // float sum_resz = mz[i] * brmsz;
            //
            // float sum_cells = (sum_resx + sum_resy + sum_resz);
    float sum_cells = loopcells(mx, my, mz, brmsx, brmsy, brmsz, i);
    float dt = delta_time;

    // Second summatory
    sn[i] += sin(wc_val * ctime) * sum_cells * dt; // dt está mal, hay que usar la fracción de dt correspondiente a cada micropaso de rk45
    cn[i] += cos(wc_val * ctime) * sum_cells * dt;

    float PREFACTOR = (gammaLL * nspins_val) / (2 * PI); // PREFACTOR = gammaLL * N
    float G = PREFACTOR * (amul(sn, cos(wc_val * ctime), i) - amul(cn, sin(wc_val * ctime), i));
    //float G = PREFACTOR * (sn[i]* cos(wc_val * ctime) - cn[i]* sin(wc_val * ctime));

    //float3 brms = {brms_x[i], brms_y[i], brms_z[i]};
    float new_term_x = brmsx * G;
    float new_term_y = brmsy * G;
    float new_term_z = brmsz * G;

    // Beff = Beff - new_term
    tx[i] -= new_term_x;
    ty[i] -= new_term_y;
    tz[i] -= new_term_z;

}
