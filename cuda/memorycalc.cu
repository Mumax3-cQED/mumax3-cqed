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
// __inline__ __device__ float warpReduce(float val) {
//
//     for (int offset = warpSize/2; offset > 0; offset /= 2) {
//         val += __shfl_down_sync(0xffffffff, val, offset);
//     }
//
//     return val;
// }
//
// __inline __device__ void sumKernel(float* d_data, int size)
// {
//     int tid = threadIdx.x + blockDim.x * blockIdx.x;
//     for (int s = blockDim.x / 2; s > 0; s >>= 1)
//     {
//         if (tid < size && tid + s < size)
//         {
//             d_data[tid] += d_data[tid + s];
//         }
//         __syncthreads();
//     }
// }
//
//
// __inline__ __device__ float loopcells(float* mx, float* my, float* mz, float brmsx, float brmsy, float brmsz, int idx, int n) {
//
//   // float3 mm = {mx[idx], my[idx], mz[idx]};
//   //     float3 bbrms = {brmsx[idx], brmsy[idx], brmsz[idx]};
//   //     float sum_res = dot(mm, bbrms);
//
//       //  sum_res = warpReduce(sum_res);
//       // return sum_res;
//     float sum_resx = amul(mx, brmsx, idx);
//     float sum_resy = amul(my, brmsy, idx);
//     float sum_resz = amul(mz, brmsz, idx);
//
//     // Use warp-synchronous programming to sum results across threads
//   sum_resx = warpReduce(sum_resx);
//   sum_resy = warpReduce(sum_resy);
//   sum_resz = warpReduce(sum_resz);
//
//     return (sum_resx + sum_resy + sum_resz);
// }
//
// __inline__ __device__ void sum3D(float* data, int dim_x, int dim_y, int dim_z, float* result) {
//     int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
//     int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
//     int tid_z = blockIdx.z * blockDim.z + threadIdx.z;
//     float sum = 0.0f;
//     for (int i = tid_x; i < dim_x; i += blockDim.x * gridDim.x) {
//         for (int j = tid_y; j < dim_y; j += blockDim.y * gridDim.y) {
//             for (int k = tid_z; k < dim_z; k += blockDim.z * gridDim.z) {
//                 sum += data[i + j * dim_x + k * dim_x * dim_y];
//             }
//         }
//     }
//     atomicAdd(result, sum);
// }

// Memory term
extern "C" __global__ void
memcalc(float* __restrict__ snx,   float* __restrict__ sny, float* __restrict__ snz,
        float* __restrict__ cnx, float* __restrict__ cny, float* __restrict__ cnz,
        float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
        float* __restrict__ wc, float wc_mul,
        float* __restrict__ brms_x, float brmsx_mul,
        float* __restrict__ brms_y, float brmsy_mul,
        float* __restrict__ brms_z, float brmsz_mul,
        float delta_time, float ctime, int Nx, int Ny, int Nz){//,  uint8_t PBC) {
        //float delta_time, float ctime, float delta_vol, int Nx, int Ny, int Nz, uint8_t PBC) {

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    if (ix >= Nx || iy >= Ny || iz >= Nz) {
       return;
    }

    int i = idx(ix, iy, iz);

    float wc_val = amul(wc, wc_mul, i);

    float brmsx = amul(brms_x, brmsx_mul, i);
    float brmsy = amul(brms_y, brmsy_mul, i);
    float brmsz = amul(brms_z, brmsz_mul, i);

    float dt = delta_time;

    // Second summatory
    snx[i] += sin(wc_val * ctime) *  amul(mx, brmsx, i) * dt;
    sny[i] += sin(wc_val * ctime) *  amul(my, brmsy, i) * dt;
    snz[i] += sin(wc_val * ctime) *  amul(mz, brmsz, i) * dt;

    cnx[i] += cos(wc_val * ctime) *  amul(mx, brmsx, i) * dt;
    cny[i] += cos(wc_val * ctime) *  amul(my, brmsy, i) * dt;
    cnz[i] += cos(wc_val * ctime) *  amul(mz, brmsz, i) * dt;


// sum3D(snx, Nx, Ny, Nz, snx);
// sum3D(sny, Nx, Ny, Nz, sny);
// sum3D(snz, Nx, Ny, Nz, snz);
//
// sum3D(cnx, Nx, Ny, Nz, cnx);
// sum3D(cny, Nx, Ny, Nz, cny);
// sum3D(cnz, Nx, Ny, Nz, cnz);

    // int n = Nx * Ny * Nz;
    // sumKernel(snx, n);
    // sumKernel(sny, n);
    // sumKernel(snz, n);
    //
    // sumKernel(cnx, n);
    // sumKernel(cny, n);
    // sumKernel(cnz, n);
}
