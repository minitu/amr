//#include "wr.h"
#include <cub/cub.cuh>
#include <math_constants.h>
#include <math_functions.h>
#include <stdio.h>
#include <cfloat>

#define USE_GPUMANAGER 0
#define SUB_BLOCK_SIZE 8
#define NUM_DIMS 3

#define gpuSafe(retval) gpuPrintErr((retval), __FILE__, __LINE__)
#define gpuCheck() gpuPrintErr(cudaGetLastError(), __FILE__, __LINE__)

inline void gpuPrintErr(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess)
    fprintf(stderr,"CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
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

__global__ void decisionKernel1(float *u, float *delu, float *delua, float dx, float dy, float dz, int block_size) {
#define INDEX(i,j,k) (((k) * (block_size+2) + (j)) * (block_size+2) + (i))
#define INDEX4(d,i,j,k) ((((d) * (block_size+2) + (k)) * (block_size+2) + (j)) * (block_size+2) + (i))
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
  if (((gx >= 1 && gx <= block_size) && (gy >= 1 && gy <= block_size)) && (gz >= 1 && gz <= block_size)) {
    // d/dx
    u_pos = (tx < SUB_BLOCK_SIZE-1) ? (u_s[tx+1][ty][tz]) : (u[INDEX(gx+1,gy,gz)]);
    u_neg = (tx > 0) ? (u_s[tx-1][ty][tz]) : (u[INDEX(gx-1,gy,gz)]);
    delu[INDEX4(0,gx,gy,gz)] = (u_pos - u_neg)*delx;
    delua[INDEX4(0,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*delx;

    // d/dy
    u_pos = (ty < SUB_BLOCK_SIZE-1) ? (u_s[tx][ty+1][tz]) : (u[INDEX(gx,gy+1,gz)]);
    u_neg = (ty > 0) ? (u_s[tx][ty-1][tz]) : (u[INDEX(gx,gy-1,gz)]);
    delu[INDEX4(1,gx,gy,gz)] = (u_pos - u_neg)*dely;
    delua[INDEX4(1,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*dely;

    // d/dz
    u_pos = (tz < SUB_BLOCK_SIZE-1) ? (u_s[tx][ty][tz+1]) : (u[INDEX(gx,gy,gz+1)]);
    u_neg = (tz > 0) ? (u_s[tx][ty][tz-1]) : (u[INDEX(gx,gy,gz-1)]);
    delu[INDEX4(2,gx,gy,gz)] = (u_pos - u_neg)*delz;
    delua[INDEX4(2,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*delz;
  }
#undef INDEX
#undef INDEX4
}

__global__ void decisionKernel2(float *delu, float *delua, float *delu_n_g, float *error_g, float *errors_g, float refine_filter, float dx, float dy, float dz, int block_size) {
#define INDEX(i,j,k) (((k) * (block_size+2) + (j)) * (block_size+2) + (i))
#define INDEX4(d,i,j,k) ((((d) * (block_size+2) + (k)) * (block_size+2) + (j)) * (block_size+2) + (i))
#define INDEX4T(i,j,k,d) ((((d) * (block_size-2) + (k)) * (block_size-2) + (j)) * (block_size-2) + (i))
#define ERR_INDEX(i,j,k) ((((k)-2) * (block_size-2) + ((j)-2)) * (block_size-2) + ((i)-2))
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
  int gx = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int gy = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int gz = blockDim.z * blockIdx.z + threadIdx.z + 1;

  // set shared error to 0
  if (tx == 0 && ty == 0 && tz == 0)
    error_s = 0.;

  // read delu & delua into shared memory
  if (gx <= block_size && gy <= block_size && gz <= block_size) {
    for (int d = 0; d < NUM_DIMS; d++) {
      delu_s[d][tx][ty][tz] = delu[INDEX4(d,gx,gy,gz)];
      delua_s[d][tx][ty][tz] = delua[INDEX4(d,gx,gy,gz)];
    }
  }
  __syncthreads();

  // calculate error per thread
  float delu_pos, delu_neg;
  float delua_pos, delua_neg;
  float num = 0., denom = 0.;
  float error;
  if ((gx > 1 && gx < block_size) && (gy > 1 && gy < block_size) && (gz > 1 && gz < block_size)) {
    for (int d = 0; d < NUM_DIMS; d++) {
      /*
      delu_pos = (tx < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx+1][ty][tz]) : (delu[INDEX4(d,gx+1,gy,gz)]);
      delu_neg = (tx > 0) ? (delu_s[d][tx-1][ty][tz]) : (delu[INDEX4(d,gx-1,gy,gz)]);
      delua_pos = (tx < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx+1][ty][tz]) : (delua[INDEX4(d,gx+1,gy,gz)]);
      delua_neg = (tx > 0) ? (delua_s[d][tx-1][ty][tz]) : (delua[INDEX4(d,gx-1,gy,gz)]);
      */
      delu_pos = delu[INDEX4(d,gx+1,gy,gz)];
      delu_neg = delu[INDEX4(d,gx-1,gy,gz)];
      delua_pos = delua[INDEX4(d,gx+1,gy,gz)];
      delua_neg = delua[INDEX4(d,gx-1,gy,gz)];
      delu_n[0][3*d+0] = (delu_pos - delu_neg)*delx;
      delu_n[1][3*d+0] = (fabsf(delu_pos) + fabsf(delu_neg))*delx;
      delu_n[2][3*d+0] = (delua_pos + delua_neg)*delx;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,3*d+0)] = (delu_pos - delu_neg)*delx;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,9+3*d+0)] = (fabsf(delu_pos) + fabsf(delu_neg))*delx;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,18+3*d+0)] = (delua_pos + delua_neg)*delx;

      /*
      delu_pos = (ty < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx][ty+1][tz]) : (delu[INDEX4(d,gx,gy+1,gz)]);
      delu_neg = (ty > 0) ? (delu_s[d][tx][ty-1][tz]) : (delu[INDEX4(d,gx,gy-1,gz)]);
      delua_pos = (ty < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx][ty+1][tz]) : (delua[INDEX4(d,gx,gy+1,gz)]);
      delua_neg = (ty > 0) ? (delua_s[d][tx][ty-1][tz]) : (delua[INDEX4(d,gx,gy-1,gz)]);
      */
      delu_pos = delu[INDEX4(d,gx,gy+1,gz)];
      delu_neg = delu[INDEX4(d,gx,gy-1,gz)];
      delua_pos = delua[INDEX4(d,gx,gy+1,gz)];
      delua_neg = delua[INDEX4(d,gx,gy-1,gz)];
      delu_n[0][3*d+1] = (delu_pos - delu_neg)*dely;
      delu_n[1][3*d+1] = (fabsf(delu_pos) + fabsf(delu_neg))*dely;
      delu_n[2][3*d+1] = (delua_pos + delua_neg)*dely;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,3*d+1)] = (delu_pos - delu_neg)*dely;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,9+3*d+1)] = (fabsf(delu_pos) + fabsf(delu_neg))*dely;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,18+3*d+1)] = (delua_pos + delua_neg)*dely;

      /*
      delu_pos = (tz < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx][ty][tz+1]) : (delu[INDEX4(d,gx,gy,gz+1)]);
      delu_neg = (tz > 0) ? (delu_s[d][tx][ty][tz-1]) : (delu[INDEX4(d,gx,gy,gz-1)]);
      delua_pos = (tz < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx][ty][tz+1]) : (delua[INDEX4(d,gx,gy,gz+1)]);
      delua_neg = (tz > 0) ? (delua_s[d][tx][ty][tz-1]) : (delua[INDEX4(d,gx,gy,gz-1)]);
      */
      delu_pos = delu[INDEX4(d,gx,gy,gz+1)];
      delu_neg = delu[INDEX4(d,gx,gy,gz-1)];
      delua_pos = delua[INDEX4(d,gx,gy,gz+1)];
      delua_neg = delua[INDEX4(d,gx,gy,gz-1)];
      delu_n[0][3*d+2] = (delu_pos - delu_neg)*delz;
      delu_n[1][3*d+2] = (fabsf(delu_pos) + fabsf(delu_neg))*delz;
      delu_n[2][3*d+2] = (delua_pos + delua_neg)*delz;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,3*d+2)] = (delu_pos - delu_neg)*delz;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,9+3*d+2)] = (fabsf(delu_pos) + fabsf(delu_neg))*delz;
      delu_n_g[INDEX4T(gx-2,gy-2,gz-2,18+3*d+2)] = (delua_pos + delua_neg)*delz;
    }

    for (int dd = 0; dd < NUM_DIMS * NUM_DIMS; dd++) {
      num = num + pow(delu_n[0][dd], 2.);
      denom = denom + pow(delu_n[1][dd], 2.) + (refine_filter * delu_n[2][dd]) * 2;
    }
    //if (gx == 11 && gy == 11 && gz == 11)
      //printf("D [%d][%d][%d] num: %.20f, denom: %.20f, num/denom: %.20f\n", gx, gy, gz, num, denom, num/denom);

    if (denom == 0. && num != 0.) {
      //error = CUDART_INF_F;
      printf("D denom is zero!!!!!!!!!!!!!!!!!!!\n");
      error = FLT_MAX;
    }
    else if (denom != 0.0) {
      error = fmaxf(error, num/denom);
    }
    //printf("D [%d][%d][%d] adding error: %f\n", gx, gy, gz, error);

    //atomicAdd(error_g, error);

    // store error in global memory
    errors_g[ERR_INDEX(gx,gy,gz)] = error;

    // find max error in thread block
    //atomicMax(&error_s, error);
  }
  __syncthreads();

  // find max error among all thread blocks
  //if (tx == 0 && ty == 0 && tz == 0)
  //  atomicMax(error_g, error_s);

#undef INDEX
#undef INDEX4
#undef INDEX4T
#undef ERR_INDEX
}

float invokeDecisionKernel(float *u, float *delu_n, float refine_filter, float dx, float dy, float dz, int block_size) {
  float error;
#if !USE_GPUMANAGER
  float *h_error;
  float *h_delu_n;
  size_t delu_n_size = sizeof(float)*(block_size-2)*(block_size-2)*(block_size-2)*3*9;
  gpuSafe(cudaMallocHost(&h_error, sizeof(float)));
  gpuSafe(cudaMallocHost(&h_delu_n, delu_n_size));

  cudaStream_t decisionStream;
  float *d_error, *d_errors;
  float *d_u, *d_delu, *d_delua;
  float *d_delu_n;
  size_t error_size = sizeof(float)*(block_size-2)*(block_size-2)*(block_size-2);
  size_t u_size = sizeof(float)*(block_size+2)*(block_size+2)*(block_size+2);
  size_t delu_size = NUM_DIMS * u_size;

  gpuSafe(cudaStreamCreate(&decisionStream));
  gpuSafe(cudaMalloc(&d_u, u_size));
  gpuSafe(cudaMalloc(&d_delu, delu_size));
  gpuSafe(cudaMalloc(&d_delua, delu_size));
  gpuSafe(cudaMalloc(&d_delu_n, delu_n_size));
  gpuSafe(cudaMalloc(&d_error, sizeof(float)));
  gpuSafe(cudaMemset(d_error, 0, sizeof(float)));
  gpuSafe(cudaMalloc(&d_errors, error_size));

  gpuSafe(cudaMemcpyAsync(d_u, u, u_size, cudaMemcpyHostToDevice, decisionStream));

  int sub_block_cnt = ceil((float)(block_size+2)/SUB_BLOCK_SIZE);
  dim3 dimGrid(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  dim3 dimBlock(SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE);
  decisionKernel1<<<dimGrid, dimBlock, 0, decisionStream>>>(d_u, d_delu, d_delua, dx, dy, dz, block_size);
  gpuCheck();

  sub_block_cnt = ceil((float)block_size/SUB_BLOCK_SIZE);
  dimGrid = dim3(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  decisionKernel2<<<dimGrid, dimBlock, 0, decisionStream>>>(d_delu, d_delua, d_delu_n, d_error, d_errors, refine_filter, dx, dy, dz, block_size);
  gpuCheck();

  gpuSafe(cudaStreamSynchronize(decisionStream));
  
  // max reduction using cub
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_errors, d_error, (block_size-2)*(block_size-2)*(block_size-2));

  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_errors, d_error, (block_size-2)*(block_size-2)*(block_size-2));

  gpuSafe(cudaMemcpyAsync(h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost, decisionStream));
  gpuSafe(cudaMemcpyAsync(h_delu_n, d_delu_n, delu_n_size, cudaMemcpyDeviceToHost, decisionStream));
  
  gpuSafe(cudaStreamSynchronize(decisionStream));
 
  memcpy(delu_n, h_delu_n, delu_n_size);
  error = *h_error;

  gpuSafe(cudaFree(d_u));
  gpuSafe(cudaFree(d_delu));
  gpuSafe(cudaFree(d_delua));
  gpuSafe(cudaFree(d_delu_n));
  gpuSafe(cudaFree(d_error));
  gpuSafe(cudaFree(d_errors));
  gpuSafe(cudaFree(d_temp_storage));
  gpuSafe(cudaFreeHost(h_error));
  gpuSafe(cudaFreeHost(h_delu_n));

  gpuSafe(cudaStreamDestroy(decisionStream));

#else

#endif

  return error;
}
