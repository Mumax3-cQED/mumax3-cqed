// MODIFIED INMA
#include "stencil.h"

extern "C" __global__ void
mdatatemp(float* __restrict__ dst_sinx, float* __restrict__ dst_siny, float* __restrict__ dst_sinz,
          float* __restrict__ dst_cosx, float* __restrict__ dst_cosy, float* __restrict__ dst_cosz, float* __restrict__ current_ctime,
          float* __restrict__ delta_time, float* __restrict__ brms_x, float* __restrict__ brms_y, float* brms_z, float* __restrict__ wc_vals,
          float* __restrict__ current_mx, float* __restrict__ current_my, float* __restrict__ current_mz,
          float ctime, float h_delta, float brmsx, float brmsy, float brmsz, float wc, int Nx, int Ny, int Nz) {

        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        int iz = blockIdx.z * blockDim.z + threadIdx.z;

        if (ix >= Nx || iy >= Ny || iz >= Nz) {
            return;
        }

        // central cell
        int i = idx(ix, iy, iz);

        dst_sinx[i] += sin(wc * ctime) * current_mx[i];
        dst_siny[i] += sin(wc * ctime) * current_my[i];
        dst_sinz[i] += sin(wc * ctime) * current_mz[i];

        dst_cosx[i] += cos(wc * ctime) * current_mx[i];
        dst_cosy[i] += cos(wc * ctime) * current_my[i];
        dst_cosz[i] += cos(wc * ctime) * current_mz[i];

        current_ctime[i] = ctime;

        delta_time[i] = h_delta;
        brms_x[i] = brmsx;
        brms_y[i] = brmsy;
        brms_z[i] = brmsz;
        wc_vals[i] = wc;
}
