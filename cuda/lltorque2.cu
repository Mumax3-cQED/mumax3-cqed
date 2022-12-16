#include "amul.h"
#include "float3.h"
#include <stdint.h>
#include "constants.h"

// Landau-Lifshitz torque.
extern "C" __global__ void
lltorque2(float* __restrict__  tx, float* __restrict__  ty, float* __restrict__  tz,
          float* __restrict__  mx, float* __restrict__  my, float* __restrict__  mz,
          float* __restrict__  hx, float* __restrict__  hy, float* __restrict__  hz,
          float* __restrict__  alpha_, float alpha_mul, int N) {
          // float* __restrict__  alpha_, float alpha_mul, int hbar_factor, int N) {

    int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    if (i < N) {

        float3 m = {mx[i], my[i], mz[i]};
        float3 H = {hx[i], hy[i], hz[i]};
        float alpha = amul(alpha_, alpha_mul, i);

        //float factor = 1;//(hbar_factor == 1 ? HBAR : 1);

        float3 mxH =  cross(m, H);
        float gilb = -1.0f / (1.0f + alpha * alpha);
        // float3 torque = factor * gilb  * (mxH + alpha * cross(m, mxH));
        float3 torque = gilb  * (mxH + alpha * cross(m, mxH));

        // tx[i] = HBAR * torque.x;
        // ty[i] = HBAR * torque.y;
        // tz[i] = HBAR * torque.z;

        tx[i] = torque.x;
        ty[i] = torque.y;
        tz[i] = torque.z;
    }
}
