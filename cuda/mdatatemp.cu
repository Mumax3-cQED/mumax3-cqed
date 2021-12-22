extern "C" __global__ void
mdatatemp(float* __restrict__ dst_sinx, float* __restrict__ dst_siny, float* __restrict__ dst_sinz,
          float* __restrict__ dst_cosx, float* __restrict__ dst_cosy, float* __restrict__ dst_cosz,
          float* __restrict__ current_mx, float* __restrict__ current_my, float* __restrict__ current_mz, float ctime, float wc, int N) {

      int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

      if(i < N) {

            dst_sinx[i] += sin(wc * ctime) * current_mx[i];
            dst_siny[i] += sin(wc * ctime) * current_my[i];
            dst_sinz[i] += sin(wc * ctime) * current_mz[i];

            dst_cosx[i] += cos(wc * ctime) * current_mx[i];
            dst_cosy[i] += cos(wc * ctime) * current_my[i];
            dst_cosz[i] += cos(wc * ctime) * current_mz[i];
      }
}
