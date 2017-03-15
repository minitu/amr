//#include "wr.h"
#include <math_constants.h>

#define USE_GPUMANAGER 0
#define SUB_BLOCK_SIZE 8
#define NUM_DIMS 3

__global__ void decisionKernel1(float *u, float *delu, float *delua, float dx, float dy, float dz, int block_size) {
#define INDEX(i,j,k) ((k * (block_size+2) + j) * (block_size+2) + i)
#define INDEX4(d,i,j,k) (((d * (block_size+2) + k) * (block_size+2) + j) * (block_size+2) + i)
  float delx = 0.5/dx;
  float dely = 0.5/dy;
  float delz = 0.5/dz;
  __shared__ float u_s[SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int gx = blockDim.x * blockIdx.x + threadIdx.x;
  int gy = blockDim.y * blockIdx.y + threadIdx.y;
  int gz = blockDim.z * blockIdx.z + threadIdx.z;

  // read u into shared memory
  if ((gx < block_size + 2) && (gy < block_size + 2) && (gz < block_size + 2)) {
    u_s[tx][ty][tz] = u[INDEX(gx,gy,gz)];
  }
  __syncthreads();

  // calculate differentials
  float u_pos, u_neg;
  if (gx >= 1 && gy >= 1 && gz >= 1 && gx <= block_size && gy <= block_size && gz <= block_size) {
    // d/dx
    u_pos = (tx < SUB_BLOCK_SIZE) ? (u_s[tx+1][ty][tz]) : (u[INDEX(gx+1,gy,gz)]);
    u_neg = (tx > 0) ? (u_s[tx-1][ty][tz]) : (u[INDEX(gx-1,gy,gz)]);
    delu[INDEX4(0,gx,gy,gz)] = (u_pos - u_neg)*delx;
    delua[INDEX4(0,gx,gy,gz)] = (abs(u_pos) + abs(u_neg))*delx;

    // d/dy
    u_pos = (ty < SUB_BLOCK_SIZE) ? (u_s[tx][ty+1][tz]) : (u[INDEX(gx,gy+1,gz)]);
    u_neg = (ty > 0) ? (u_s[tx][ty-1][tz]) : (u[INDEX(gx,gy-1,gz)]);
    delu[INDEX4(1,gx,gy,gz)] = (u_pos - u_neg)*dely;
    delua[INDEX4(1,gx,gy,gz)] = (abs(u_pos) + abs(u_neg))*dely;

    // d/dz
    u_pos = (tz < SUB_BLOCK_SIZE) ? (u_s[tx][ty][tz+1]) : (u[INDEX(gx,gy,gz+1)]);
    u_neg = (tz > 0) ? (u_s[tx][ty][tz-1]) : (u[INDEX(gx,gy,gz-1)]);
    delu[INDEX4(2,gx,gy,gz)] = (u_pos - u_neg)*delz;
    delua[INDEX4(2,gx,gy,gz)] = (abs(u_pos) + abs(u_neg))*delz;
  }
#undef INDEX
#undef INDEX4
}

__device__ static float atomicMax(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_i, assumed,
        __float_as_int(fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void decisionKernel2(float *u, float *delu, float *delua, float *error_g, float refine_filter, float dx, float dy, float dz, int block_size) {
#define INDEX(i,j,k) ((k * (block_size+2) + j) * (block_size+2) + i)
#define INDEX4(d,i,j,k) (((d * (block_size+2) + k) * (block_size+2) + j) * (block_size+2) + i)
  float delx = 0.5/dx;
  float dely = 0.5/dy;
  float delz = 0.5/dz;
  float delu_n[3][NUM_DIMS * NUM_DIMS];
  __shared__ float delu_s[NUM_DIMS][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];
  __shared__ float delua_s[NUM_DIMS][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];
  __shared__ float error_s;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int gx = blockDim.x * blockIdx.x + threadIdx.x;
  int gy = blockDim.y * blockIdx.y + threadIdx.y;
  int gz = blockDim.z * blockIdx.z + threadIdx.z;

  // set shared error to 0
  if (tx == 0 && ty == 0 && tz == 0)
    error_s = 0;

  // read delu & delua into shared memory
  if ((gx >= 1 && gx <= block_size) && (gy >= 1 && gy <= block_size) && (gz >= 1 && gz <= block_size)) {
    for (int d = 0; d < NUM_DIMS; d++) {
      delu_s[d][tx][ty][tz] = delu[INDEX4(d,gx,gy,gz)];
      delua_s[d][tx][ty][tz] = delua[INDEX4(d,gx,gy,gz)];
    }
  }
  __syncthreads();

  // calculate error per thread
  float delu_pos, delu_neg;
  float delua_pos, delua_neg;
  float num = 0, denom = 0;
  float error;
  if (gx > 1 && gy > 1 && gz > 1 && gx < block_size && gy < block_size && gz < block_size) {
    for (int d = 0; d < NUM_DIMS; d++) {
      delu_pos = (tx < SUB_BLOCK_SIZE) ? (delu_s[d][tx+1][ty][tz]) : (delu[INDEX4(d,gx+1,gy,gz)]);
      delu_neg = (tx > 0) ? (delu_s[d][tx-1][ty][tz]) : (delu[INDEX4(d,gx-1,gy,gz)]);
      delua_pos = (tx < SUB_BLOCK_SIZE) ? (delua_s[d][tx+1][ty][tz]) : (delua[INDEX4(d,gx+1,gy,gz)]);
      delua_neg = (tx > 0) ? (delua_s[d][tx-1][ty][tz]) : (delua[INDEX4(d,gx-1,gy,gz)]);
      delu_n[0][3*d+0] = (delu_pos - delu_neg)*delx;
      delu_n[1][3*d+0] = (abs(delu_pos) + abs(delu_neg))*delx;
      delu_n[2][3*d+0] = (delua_pos + delua_neg)*delx;

      delu_pos = (ty < SUB_BLOCK_SIZE) ? (delu_s[d][tx][ty+1][tz]) : (delu[INDEX4(d,gx,gy+1,gz)]);
      delu_neg = (ty > 0) ? (delu_s[d][tx][ty-1][tz]) : (delu[INDEX4(d,gx,gy-1,gz)]);
      delua_pos = (ty < SUB_BLOCK_SIZE) ? (delua_s[d][tx][ty+1][tz]) : (delua[INDEX4(d,gx,gy+1,gz)]);
      delua_neg = (ty > 0) ? (delua_s[d][tx][ty-1][tz]) : (delua[INDEX4(d,gx,gy-1,gz)]);
      delu_n[0][3*d+1] = (delu_pos - delu_neg)*dely;
      delu_n[1][3*d+1] = (abs(delu_pos) + abs(delu_neg))*dely;
      delu_n[2][3*d+1] = (delua_pos + delua_neg)*dely;

      delu_pos = (tz < SUB_BLOCK_SIZE) ? (delu_s[d][tx][ty][tz+1]) : (delu[INDEX4(d,gx,gy,gz+1)]);
      delu_neg = (tz > 0) ? (delu_s[d][tx][ty][tz-1]) : (delu[INDEX4(d,gx,gy,gz-1)]);
      delua_pos = (tz < SUB_BLOCK_SIZE) ? (delua_s[d][tx][ty][tz+1]) : (delua[INDEX4(d,gx,gy,gz+1)]);
      delua_neg = (tz > 0) ? (delua_s[d][tx][ty][tz-1]) : (delua[INDEX4(d,gx,gy,gz-1)]);
      delu_n[0][3*d+2] = (delu_pos - delu_neg)*delz;
      delu_n[1][3*d+2] = (abs(delu_pos) + abs(delu_neg))*delz;
      delu_n[2][3*d+2] = (delua_pos + delua_neg)*delz;
    }

    for (int dd = 0; dd < NUM_DIMS * NUM_DIMS; dd++) {
      num += powf(delu_n[0][dd], 2.0);
      denom += powf(delu_n[1][dd], 2.0) + (refine_filter * delu_n[2][dd]) * 2;
    }

    if (denom == 0.0 && num != 0.0) {
      error = CUDART_INF_F;
    }
    else if (denom != 0.0) {
      error = fmaxf(error, num/denom);
    }

    // find max error in thread block
    atomicMax(&error_s, error);
  }
  __syncthreads();

  // find max error among all thread blocks
  if (tx == 0 && ty == 0 && tz == 0)
    atomicMax(error_g, error_s);
#undef INDEX
#undef INDEX4
}

float invokeDecisionKernel(float *u, int dx, int dy, int dz, int block_size) {
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
  cudaMalloc(&d_delua, delu_size);
  cudaMalloc(&d_error, sizeof(float));
  
  cudaMemcpyAsync(d_u, u, u_size, cudaMemcpyHostToDevice, decisionStream);

  int sub_block_cnt = ceil((float)(block_size+2)/SUB_BLOCK_SIZE);
  dim3 dimGrid(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  dim3 dimBlock(SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE);
  decisionKernel1<<<dimGrid, dimBlock, 0, decisionStream>>>(d_u, d_delu, d_delua, dx, dy, dz, block_size);

  sub_block_cnt = ceil((float)block_size/SUB_BLOCK_SIZE);
  dim3 dimGrid(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  decisionKernel2<<<dimGrid, dimBlock, 0, decisionStream>>>(d_u, d_delu, d_delua, d_error, refine_filter, dx, dy, dz, block_size);
  
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
