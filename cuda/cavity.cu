#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

// See cavity.go for more details.
extern "C" __global__
void addcavity(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
            float* __restrict__ sn, float* __restrict__ cn,
            float* __restrict__ brms_x, float* __restrict__ brms_y,  float* __restrict__ brms_z,
            float wc, float kappa, float x0, float p0, float vc2_hbar, float dt, float ctime, float brms_m, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
       return;
    }

    int i = idx(ix, iy, iz);

    // Summatory
    sn[i] += exp(kappa * ctime) * sin(wc * ctime) * brms_m * dt;
    cn[i] += exp(kappa * ctime) * cos(wc * ctime) * brms_m * dt;

    float G = exp(-kappa * ctime) * (cos(wc * ctime) * (x0 - vc2_hbar * sn[i]) - sin(wc * ctime) * (p0 - vc2_hbar * cn[i]));

    // This is the new term to Beff
    float new_term_x = brms_x[i] * G;
    float new_term_y = brms_y[i] * G;
    float new_term_z = brms_z[i] * G;

    // Beff = Beff + new_term
    tx[i] += new_term_x;
    ty[i] += new_term_y;
    tz[i] += new_term_z;
}
