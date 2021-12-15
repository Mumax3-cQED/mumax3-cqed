
extern "C" __global__ void
mdatatemp(float* __restrict__  dst_x, float* __restrict__  dst_y, float* __restrict__  dst_z,  float* __restrict__  temp_dt,
      float* __restrict__  mx_temp, float* __restrict__  my_temp, float* __restrict__  mz_temp, float dt, int N) {

    	// for (int i = 0; i < size_x*size_y*size_z; i++) {
      int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;

      if(i < N) {
    				dst_x[i] += mx_temp[i];
    				dst_y[i] += my_temp[i];
    				dst_z[i] += mz_temp[i];
    				temp_dt[i] = dt;
      }
}
