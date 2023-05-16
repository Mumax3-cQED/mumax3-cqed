// MODIFIED INMA
#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "constants.h"
#include "stencil.h"

//#define idx3d(ix,iy,iz) ( ix + iy*Nx + iz*Nx*Ny )

//__device__ float cc;
//__device__ float ss;

// Note that warpReduce is a custom function that sums the input across threads in a warp using warp-synchronous programming
// This function uses the shuffle operation (__shfl_down_sync) to perform a pairwise reduction of the input value across threads in a warp
// The warpSize constant is used to determine the number of threads in a warp
// The function returns the final reduced sum to the caller
/*
__inline__ __device__ float warpReduce(float val) {

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__inline__ __device__ float loopcells(float* mx, float* my, float* mz, float brmsx, float* brmsy, float* brmsz, int ii) {

    float3 mm = {mx[ii], my[ii], mz[ii]};
    float3 bbrms = {brmsx[ii], brmsy[ii], brmsz[ii]};
    float sum_res = dot(mm, bbrms);

    // Use warp-synchronous programming to sum results across threads
    sum_res = warpReduce(sum_res);
    //sum_resy = warpReduce(sum_resy);
    //sum_resz = warpReduce(sum_resz);

    return sum_res;//(sum_resx + sum_resy + sum_resz);
}
*/

__inline__ __device__ float warpReduce(float val) {

    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    return val;
}

__inline__ __device__ float loopcells(float* mx, float* my, float* mz, float* brms_x, float* brms_y, float* brms_z, int idx) {

    float sum_resx = mx[idx] * brms_x[idx];
    float sum_resy = my[idx] * brms_y[idx];
    float sum_resz = mz[idx] * brms_z[idx];

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
          float* __restrict__ nspins, float nspins_mul,
          float* __restrict__ brms_x, float brmsx_mul,
          float* __restrict__ brms_y, float brmsy_mul,
          float* __restrict__ brms_z, float brmsz_mul,
          float delta_time, float ctime, float gamma_val, int Nx, int Ny, int Nz) {
        //float delta_time, float ctime, float delta_vol, int Nx, int Ny, int Nz, uint8_t PBC) {

        int ix = blockIdx.x * blockDim.x + threadIdx.x;
        int iy = blockIdx.y * blockDim.y + threadIdx.y;
        int iz = blockIdx.z * blockDim.z + threadIdx.z;

        if (ix >= Nx || iy >= Ny || iz >= Nz) {
           return;
        }

        int i = idx(ix, iy, iz);
        //int i = idx3d(ix, iy, iz);

        float wc_val = amul(wc, wc_mul, i);//amul(wc, wc_mul, i);
        float nspins_val = amul(nspins, nspins_mul, i);//amul(nspins, nspins_mul, i);

        float brmsx = amul(brms_x, brmsx_mul, i);
        float brmsy = amul(brms_y, brmsy_mul, i);
        float brmsz = amul(brms_z, brmsz_mul, i);

        //
        // float brmsx = brms_x[i] * brmsx_mul;//amul(brms_x, brmsx_mul, i);
        // float brmsy = brms_y[i]* brmsy_mul;//amul(brms_y, brmsy_mul, i);
        // float brmsz = brms_z[i]* brmsz_mul;//amul(brms_z, brmsz_mul, i);

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

        //float sum_cells = loopcells(mx, my, mz, brms_x, brms_y, brms_z, i);
        float sum_cells = amul(mx, brmsx, i) + amul(my, brmsy, i) + amul(mz, brmsz, i);
        float dt = delta_time;

        // Second summatory
        sn[i] += sin(wc_val * ctime) * sum_cells * dt; // dt está mal, hay que usar la fracción de dt correspondiente a cada micropaso de rk45
        cn[i] += cos(wc_val * ctime) * sum_cells * dt;

        float PREFACTOR = nspins_val* GAMMA0; // PREFACTOR = gammaLL * N
        float gamma = PREFACTOR * (amul(sn, cos(wc_val * ctime), i) - amul(cn, sin(wc_val * ctime), i));

        // float term_x = amul(brms_x, gamma, i);
        // float term_y = amul(brms_y, gamma, i);
        // float term_z = amul(brms_z, gamma, i);

        float3 brms = {brmsx, brmsy, brmsz};
        float3 new_term = brms * gamma;

        // Beff - new_term
        tx[i] -= new_term.x;
        ty[i] -= new_term.y;
        tz[i] -= new_term.z;

/*
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
       return;
    }
//printf("pasa");
    // int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int idy = blockDim.y * blockIdx.y + threadIdx.y;
    // int idz = blockDim.z * blockIdx.z + threadIdx.z;
    int i = idx(ix, iy, iz);
    // //int i = idx3d(ix, iy, iz);
    //
    // if (idx >= Nx || idy >= Ny || idz >= Nz) {
    //    return;
    // }
    // int idx =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    //
    // if (idx < N) {
            //int index = idx + Nx * (idy + Ny * idz);
            //int index = idx(idx, idy, idz);


   float wc_val = amul(wc, wc_mul, idx);
    float nspins_val = amul(nspins, nspins_mul, idx);

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

    // float sum_cells = mx[i] * brms_x[i] + my[i] * brms_y[i] + mz[i] * brms_z[i];
    float sum_cells = loopcells(mx, my, mz, brms_x, brms_y, brms_z, i);
    float dt = delta_time;

    // Second summatory
    sn[i] += sin(amul(wc, ctime, i)) * sum_cells * dt; // dt está mal, hay que usar la fracción de dt correspondiente a cada micropaso de rk45
    cn[i] += cos(amul(wc, ctime, i)) * sum_cells * dt;

    float PREFACTOR = amul(nspins, gamma_val, i); // PREFACTOR = gammaLL * N
    float gamma = PREFACTOR * (amul(sn, cos(amul(wc, ctime, i)), i) - (amul(cn, sin(amul(wc, ctime, i)), i)));

    float term_x = amul(brms_x, gamma, i);
    float term_y = amul(brms_y, gamma, i);
    float term_z = amul(brms_z, gamma, i);

    // Beff - new_term
    tx[i] -= term_x;
    ty[i] -= term_y;
    tz[i] -= term_z;
*/
    //float3 brms = {brmsx, brmsy, brmsz};
    // float new_term = (PREFACTOR * dot(mm, brms) * dt); //* gamma;
    // float term_x = amul(brms_x, new_term, i);
    // float term_y = amul(brms_y, new_term, i);
    // float term_z = amul(brms_z, new_term, i);
//printf("nspins %f \n", nspins_val);

//float wc_val = amul(wc, wc_mul, i);
//float nspins_val = amul(nspins, nspins_mul, i);

// float brmsx = amul(brms_x, brmsx_mul, i);
// float brmsy = amul(brms_y, brmsy_mul, i);
// float brmsz = amul(brms_z, brmsz_mul, i);
/*
float t_n = ctime;// - delta_time;
float omega_c_t_n = wc[i] * t_n;
float dt = delta_time;
//float sum_b_rms_M = mx[i] * brms_x[i] + my[i] * brms_y[i] + mz[i] * brms_z[i];
float sum_b_rms_M = loopcells(mx, my, mz, brms_x, brms_y, brms_z, i);
sn[index] += sin(omega_c_t_n) * sum_b_rms_M * dt;
cn[index] += cos(omega_c_t_n) * sum_b_rms_M * dt;
//float PREFACTOR = nspins_val * gamma_val; // PREFACTOR = gammaLL * N
//printf("llega");
float gamma_t_n = gamma_val * nspins[i] *(cos(omega_c_t_n) * sn[i] - sin(omega_c_t_n) * cn[i]);
float3 brms = {brms_x[i], brms_y[i], brms_z[i] };
float3 new_term = brms * gamma_t_n;
tx[index] -= new_term.x;
ty[index] -= new_term.y;
tz[index] -= new_term.z;
*/
 // }

}
