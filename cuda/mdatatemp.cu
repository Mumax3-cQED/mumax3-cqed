extern "C" __global__ void
mdatatemp(float* __restrict__  dst,
      float* __restrict__  mx_temp, float* __restrict__  my_temp, float* __restrict__  mz_temp, float dt, int size_x, int size_y, int size_z, int N) {

      int i =  ( blockIdx.y*gridDim.x + blockIdx.x ) * blockDim.x + threadIdx.x;
    //
      if(i < N) {

      // int idx_cell = 0;

    	for (int z = 0; z < size_z; z++) {
    		for (int y = 0; y < size_y; y++) {
    			for (int x = 0; x < size_x; x++) {

    				float arr[5];

    				arr[0] = (float)i;
    				arr[1] = mx_temp[x];
    				arr[2] = my_temp[y];
    				arr[3] = mz_temp[z];
    				arr[4] = dt;

    			  dst[i] = *arr;

    				// idx_cell += 1;
    			}
    		}
      }
    }
}
