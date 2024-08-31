#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

// See cavity.go for more details.
extern "C" __global__
void addcavity(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
            float* __restrict__ wc, float wc_mul,
            float* __restrict__ kappa, float kappa_mul,
            float* __restrict__ brms_x, float* __restrict__ brms_y,  float* __restrict__ brms_z,
            float x0, float p0, float msat_cell, float ctime, float sn, float cn, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
       return;
    }

    int i = idx(ix, iy, iz);

    float wc_val = amul(wc, wc_mul, i);

    float kappa_val = amul(kappa, kappa_mul, i);

    float G = exp(-kappa_val * ctime) * (cos(wc_val * ctime) * (x0 - msat_cell * sn) - sin(wc_val * ctime) * (p0 - msat_cell * cn));

    // This is the new term to Beff
    float new_term_x = brms_x[i] * G;
    float new_term_y = brms_y[i] * G;
    float new_term_z = brms_z[i] * G;

    // Beff = Beff + new_term
    tx[i] += new_term_x;
    ty[i] += new_term_y;
    tz[i] += new_term_z;
}
