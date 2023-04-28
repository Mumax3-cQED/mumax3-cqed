// MODIFIED INMA
#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "constants.h"
#include "stencil.h"

//#define idx3d(ix,iy,iz) ( ix + iy*Nx + iz*Nx*Ny )

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

    float sum_resx = mx[idx] * brmsx;
    float sum_resy = my[idx] * brmsy;
    float sum_resz = mz[idx] * brmsz;

    // Use warp-synchronous programming to sum results across threads
    sum_resx = warpReduce(sum_resx);
    sum_resy = warpReduce(sum_resy);
    sum_resz = warpReduce(sum_resz);

    return sum_resx + sum_resy + sum_resz;
}

// Bcustom amount calculation.
extern "C" __global__ void
calcspinbeff(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__ sn, float* __restrict__ cn,
          float* __restrict__ wc, float wc_mul,
          float* __restrict__ msat, float msat_mul,
          float* __restrict__ brms_x, float brmsx_mul,
          float* __restrict__ brms_y, float brmsy_mul,
          float* __restrict__ brms_z, float brmsz_mul,
          float delta_time, float ctime, float delta_vol, int Nx, int Ny, int Nz, uint8_t PBC) {

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
    float sum_cells = loopcells(mx, my, mz, brmsx, brmsy, brmsz, i);
    float dt = delta_time/GAMMA0;

    // Second summatory
    sn[i] += sin(wc_val * ctime) * sum_cells * dt;
    cn[i] += cos(wc_val * ctime) * sum_cells * dt;

    float PREFACTOR = (2 / HBAR) * delta_vol * msat_val;
    float gamma = PREFACTOR * ((cos(wc_val * ctime) * sn[i]) - (sin(wc_val * ctime) * cn[i]));

    float3 brms = {brmsx, brmsy, brmsz};
    float3 bext = brms * gamma;

    // Beff - Bcustom
    tx[i] -= bext.x;
    ty[i] -= bext.y;
    tz[i] -= bext.z;

    // Beff + Bcustom
    // tx[i] += bext.x;
    // ty[i] += bext.y;
    // tz[i] += bext.z;
}
