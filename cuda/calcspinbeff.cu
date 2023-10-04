// CREATED AND MODIFIED INMA
#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "stencil.h"

//Calculations for extra term in Beff
extern "C" __global__ void
calcspinbeff(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
            float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
            float* __restrict__ snx,   float* __restrict__ sny, float* __restrict__ snz,
            float* __restrict__ cnx, float* __restrict__ cny, float* __restrict__ cnz,
            float* __restrict__ wc, float wc_mul,
            float* __restrict__ nspins, float nspins_mul,
            float* __restrict__ brms_x, float brmsx_mul,
            float* __restrict__ brms_y, float brmsy_mul,
            float* __restrict__ brms_z, float brmsz_mul,
            float dt, float ctime, float gammaLL, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
       return;
    }

    int i = idx(ix, iy, iz);

    float wc_val = amul(wc, wc_mul, i);
    float nspins_val = amul(nspins, nspins_mul, i);

    float brmsx = amul(brms_x, brmsx_mul, i);
    float brmsy = amul(brms_y, brmsy_mul, i);
    float brmsz = amul(brms_z, brmsz_mul, i);

    // Summatory
    snx[i] += sin(wc_val * ctime) * amul(mx, brmsx, i) * dt;
    sny[i] += sin(wc_val * ctime) * amul(my, brmsy, i) * dt;
    snz[i] += sin(wc_val * ctime) * amul(mz, brmsz, i) * dt;

    cnx[i] += cos(wc_val * ctime) * amul(mx, brmsx, i) * dt;
    cny[i] += cos(wc_val * ctime) * amul(my, brmsy, i) * dt;
    cnz[i] += cos(wc_val * ctime) * amul(mz, brmsz, i) * dt;

    // Summatory
    float sn = snx[i] + sny[i] + snz[i];
    float cn = cnx[i] + cny[i] + cnz[i];

    float PREFACTOR = gammaLL * nspins_val;
    float G = PREFACTOR * (sn * cos(wc_val * ctime) - cn * sin(wc_val * ctime));

    // This is the new term to Beff
    float new_term_x = brmsx * G;
    float new_term_y = brmsy * G;
    float new_term_z = brmsz * G;

    // Beff = Beff - new_term
    tx[i] -= new_term_x;
    ty[i] -= new_term_y;
    tz[i] -= new_term_z;
}
