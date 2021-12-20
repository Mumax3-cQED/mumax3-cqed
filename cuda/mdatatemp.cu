extern "C" __global__ void
mdatatemp(float* __restrict__  dst_x, float* __restrict__  dst_y, float* __restrict__  dst_z, float* __restrict__  dst_mx_current, float* __restrict__  dst_my_current, float* __restrict__  dst_mz_current,
          float* __restrict__  mx_temp, float* __restrict__  my_temp, float* __restrict__  mz_temp,
          float* __restrict__  dst_current_mx, float* __restrict__  dst_current_my, float* __restrict__  dst_current_mz, float prevTime, float wc, float ctime, int N) {

      int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

      if(i < N) {

            dst_x[i] += sin(wc * (prevTime - ctime)) * mx_temp[i];
            dst_y[i] += sin(wc * (prevTime - ctime)) * my_temp[i];
            dst_z[i] += sin(wc * (prevTime - ctime)) * mz_temp[i];

            dst_mx_current[i] = dst_current_mx[i];
            dst_my_current[i] = dst_current_my[i];
            dst_mz_current[i] = dst_current_mz[i];

          __syncthreads();
      }
}
