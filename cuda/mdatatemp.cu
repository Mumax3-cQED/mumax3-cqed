
extern "C" __global__ void
mdatatemp(float* __restrict__  dst_x, float* __restrict__  dst_y, float* __restrict__  dst_z, float* __restrict__ sin_full_time,
      float* __restrict__  mx_temp, float* __restrict__  my_temp, float* __restrict__  mz_temp, float wc, float full_tau, float dt, int N) {

    	// for (int i = 0; i < size_x*size_y*size_z; i++) {
      int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

      if(i < N) {

    				dst_x[i] += mx_temp[i];
    				dst_y[i] += my_temp[i];
    				dst_z[i] += mz_temp[i];
            sin_full_time[i] = sin(wc * (full_tau - dt));
      }
}
