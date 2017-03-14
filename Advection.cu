#include "wr.h"
#define SUB_BLOCK_SIZE 8

__global__ void decisionKernel1(float *u, float *delu, float *delua, float *error, float dx, float dy, float dz, int block_size) {
#define INDEX(i,j,k) ((k * (block_size+2) + j) * (block_size+2) + i)
#define INDEX4(d,i,j,k) (((d * (block_size+2) + k) * (block_size+2) + j) * (block_size+2) + i)
  float delx = 0.5/dx;
  float dely = 0.5/dy;
  float delz = 0.5/dz;
  //__shared__ float delu_s[3][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];
  //__shared__ float delua_s[3][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int gx = blockDim.x * blockIdx.x + threadIdx.x;
  int gy = blockDim.y * blockIdx.y + threadIdx.y;
  int gz = blockDim.z * blockIdx.z + threadIdx.z;

  // thread (gx,gy,gz) handles point (gx+1,gy+1,gz+1)
  if (gx < block_size && gy < block_size && gz < block_size) {
    // d/dx
    delu_s[0][tx][ty][tz] = (u[INDEX(gx+2,gy+1,gz+1)] - u[INDEX(gx,gy+1,gz+1)])*delx;
    delua_s[0][tx][ty][tz] = (abs(u[INDEX(gx+2,gy+1,gz+1)]) + abs(u[INDEX(gx,gy+1,gz+1)]))*delx;

    // d/dy
    delu_s[1][tx][ty][tz] = (u[INDEX(gx+1,gy+2,gz+1)] - u[INDEX(gx+1,gy,gz+1)])*dely;
    delua_s[1][tx][ty][tz] = (abs(u[INDEX(gx+1,gy+2,gz+1)]) + abs(u[INDEX(gx+1,gy+2,gz+1)]))*dely;

    // d/dz
    delu_s[2][tx][ty][tz] = (u[INDEX(gx+1,gy+1,gz+2)] - u[INDEX(gx+1,gy+1,gz+2)])*delz;
    delua_s[2][tx][ty][tz] = (abs(u[INDEX(gx+1,gy+1,gz+2)]) + abs(u[INDEX(gx+1,gy+1,gz+2)]))*delz;
  }
#undef INDEX
#undef INDEX4
}

float Advection::invokeDecisionKernel(int block_size) {
  float error;
#ifndef USE_GPUMANAGER
  cudaStream_t decisionStream;
  float *d_error;
  float *d_u;
  float *d_delu, *d_delua;
  size_t u_size = (block_size+2)*(block_size+2)*(block_size+2);
  size_t delu_size = 3 * u_size;

  cudaStreamCreate(&decisionStream);

  cudaMalloc(&d_u, u_size);
  cudaMalloc(&d_delu, delu_size);
  cudaMalloc(&d_delua, delua_size);
  cudaMalloc(&d_error, sizeof(float));
  
  cudaMemcpyAsync(d_u, u, u_size, cudaMemcpyHostToDevice, decisionStream);

  int sub_block_cnt = ceil((float)block_size/SUB_BLOCK_SIZE);
  dim3 dimGrid(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  dim3 dimBlock(SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE);
  decisionKernel1<<<dimGrid, dimBlock, 0, decisionStream>>>(d_u, d_delu, d_delua, d_error, dx, dy, dz, block_size);
  decisionKernel2<<<dimGrid, dimBlock, 0, decisionStream>>>(d_u, d_delu, d_delua, d_error, dx, dy, dz, block_size);
  
  cudaMemcpyAsync(&error, d_error, sizeof(float), cudaMemcpyDeviceToHost, decisionStream);

  cudaStreamSynchronize(decisionStream);

  cudaFree(d_u);
  cudaFree(d_delu);
  cudaFree(d_delua);
  cudaFree(d_error);

  cudaStreamDestroy(decisionStream);
#else

#endif

  return error;
}
