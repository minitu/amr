//#include "wr.h"
#include <cub/cub.cuh>
#include <math_constants.h>
#include <math_functions.h>
#include <stdio.h>
#include <cfloat>

#define USE_GPUMANAGER 0
#define USE_SHARED_MEM 1
#define SUB_BLOCK_SIZE 8
#define NUM_DIMS 3

#define gpuSafe(retval) gpuPrintErr((retval), __FILE__, __LINE__)
#define gpuCheck() gpuPrintErr(cudaGetLastError(), __FILE__, __LINE__)

inline void gpuPrintErr(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess)
    fprintf(stderr,"CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
}

__global__ void decisionKernel1(float *u, float *delu, float *delua, float dx, float dy, float dz, int block_size) {
#define INDEX(i,j,k) (((k) * (block_size+2) + (j)) * (block_size+2) + (i))
#define INDEX4(d,i,j,k) ((((d) * (block_size+2) + (k)) * (block_size+2) + (j)) * (block_size+2) + (i))
  float delx = 0.5/dx;
  float dely = 0.5/dy;
  float delz = 0.5/dz;
#if USE_SHARED_MEM
  __shared__ float u_s[SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
#endif
  int gx = blockDim.x * blockIdx.x + threadIdx.x;
  int gy = blockDim.y * blockIdx.y + threadIdx.y;
  int gz = blockDim.z * blockIdx.z + threadIdx.z;

#if USE_SHARED_MEM
  // read u into shared memory
  if ((gx < (block_size + 2)) && (gy < (block_size + 2)) && (gz < (block_size + 2))) {
    u_s[tx][ty][tz] = u[INDEX(gx,gy,gz)];
  }
  __syncthreads();
#endif

  // calculate differentials
  float u_pos, u_neg;
  if (((gx >= 1 && gx <= block_size) && (gy >= 1 && gy <= block_size)) && (gz >= 1 && gz <= block_size)) {
    // d/dx
#if USE_SHARED_MEM
    u_pos = (tx < SUB_BLOCK_SIZE-1) ? (u_s[tx+1][ty][tz]) : (u[INDEX(gx+1,gy,gz)]);
    u_neg = (tx > 0) ? (u_s[tx-1][ty][tz]) : (u[INDEX(gx-1,gy,gz)]);
#else
    u_pos = u[INDEX(gx+1,gy,gz)];
    u_neg = u[INDEX(gx-1,gy,gz)];
#endif
    delu[INDEX4(0,gx,gy,gz)] = (u_pos - u_neg)*delx;
    delua[INDEX4(0,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*delx;

    // d/dy
#if USE_SHARED_MEM
    u_pos = (ty < SUB_BLOCK_SIZE-1) ? (u_s[tx][ty+1][tz]) : (u[INDEX(gx,gy+1,gz)]);
    u_neg = (ty > 0) ? (u_s[tx][ty-1][tz]) : (u[INDEX(gx,gy-1,gz)]);
#else
    u_pos = u[INDEX(gx,gy+1,gz)];
    u_neg = u[INDEX(gx,gy-1,gz)];
#endif
    delu[INDEX4(1,gx,gy,gz)] = (u_pos - u_neg)*dely;
    delua[INDEX4(1,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*dely;

    // d/dz
#if USE_SHARED_MEM
    u_pos = (tz < SUB_BLOCK_SIZE-1) ? (u_s[tx][ty][tz+1]) : (u[INDEX(gx,gy,gz+1)]);
    u_neg = (tz > 0) ? (u_s[tx][ty][tz-1]) : (u[INDEX(gx,gy,gz-1)]);
#else
    u_pos = u[INDEX(gx,gy,gz+1)];
    u_neg = u[INDEX(gx,gy,gz-1)];
#endif
    delu[INDEX4(2,gx,gy,gz)] = (u_pos - u_neg)*delz;
    delua[INDEX4(2,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*delz;
  }
#undef INDEX
#undef INDEX4
}

#ifdef GPU_DEBUG
__global__ void decisionKernel2(float *delu, float *delua, float *delu_n_g, float *errors, float refine_filter, float dx, float dy, float dz, int block_size) {
#else
__global__ void decisionKernel2(float *delu, float *delua, float *errors, float refine_filter, float dx, float dy, float dz, int block_size) {
#endif
#define INDEX4(d,i,j,k) ((((d) * (block_size+2) + (k)) * (block_size+2) + (j)) * (block_size+2) + (i))
#ifdef GPU_DEBUG
#define INDEX4C(i,j,k,d) ((((d) * (block_size-2) + (k)) * (block_size-2) + (j)) * (block_size-2) + (i))
#endif
#define ERR_INDEX(i,j,k) ((((k)-2) * (block_size-2) + ((j)-2)) * (block_size-2) + ((i)-2))
  float delx = 0.5/dx;
  float dely = 0.5/dy;
  float delz = 0.5/dz;
  float delu_n[3][NUM_DIMS * NUM_DIMS];
#if USE_SHARED_MEM
  __shared__ float delu_s[NUM_DIMS][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];
  __shared__ float delua_s[NUM_DIMS][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];
#endif

#if USE_SHARED_MEM
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
#endif
  int gx = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int gy = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int gz = blockDim.z * blockIdx.z + threadIdx.z + 1;

#if USE_SHARED_MEM
  // read delu & delua into shared memory
  if (gx <= block_size && gy <= block_size && gz <= block_size) {
    for (int d = 0; d < NUM_DIMS; d++) {
      delu_s[d][tx][ty][tz] = delu[INDEX4(d,gx,gy,gz)];
      delua_s[d][tx][ty][tz] = delua[INDEX4(d,gx,gy,gz)];
    }
  }
  __syncthreads();
#endif

  // calculate error per thread
  float delu_pos, delu_neg;
  float delua_pos, delua_neg;
  float num = 0., denom = 0.;
  float error = 0.0f;
  if ((gx > 1 && gx < block_size) && (gy > 1 && gy < block_size) && (gz > 1 && gz < block_size)) {
    for (int d = 0; d < NUM_DIMS; d++) {
#if USE_SHARED_MEM
      delu_pos = (tx < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx+1][ty][tz]) : (delu[INDEX4(d,gx+1,gy,gz)]);
      delu_neg = (tx > 0) ? (delu_s[d][tx-1][ty][tz]) : (delu[INDEX4(d,gx-1,gy,gz)]);
      delua_pos = (tx < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx+1][ty][tz]) : (delua[INDEX4(d,gx+1,gy,gz)]);
      delua_neg = (tx > 0) ? (delua_s[d][tx-1][ty][tz]) : (delua[INDEX4(d,gx-1,gy,gz)]);
#else
      delu_pos = delu[INDEX4(d,gx+1,gy,gz)];
      delu_neg = delu[INDEX4(d,gx-1,gy,gz)];
      delua_pos = delua[INDEX4(d,gx+1,gy,gz)];
      delua_neg = delua[INDEX4(d,gx-1,gy,gz)];
#endif
      delu_n[0][3*d+0] = (delu_pos - delu_neg)*delx;
      delu_n[1][3*d+0] = (fabsf(delu_pos) + fabsf(delu_neg))*delx;
      delu_n[2][3*d+0] = (delua_pos + delua_neg)*delx;
#ifdef GPU_DEBUG
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,3*d+0)] = delu_n[0][3*d+0];
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,9+3*d+0)] = delu_n[1][3*d+0];
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,18+3*d+0)] = delu_n[2][3*d+0];
#endif

#if USE_SHARED_MEM
      delu_pos = (ty < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx][ty+1][tz]) : (delu[INDEX4(d,gx,gy+1,gz)]);
      delu_neg = (ty > 0) ? (delu_s[d][tx][ty-1][tz]) : (delu[INDEX4(d,gx,gy-1,gz)]);
      delua_pos = (ty < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx][ty+1][tz]) : (delua[INDEX4(d,gx,gy+1,gz)]);
      delua_neg = (ty > 0) ? (delua_s[d][tx][ty-1][tz]) : (delua[INDEX4(d,gx,gy-1,gz)]);
#else
      delu_pos = delu[INDEX4(d,gx,gy+1,gz)];
      delu_neg = delu[INDEX4(d,gx,gy-1,gz)];
      delua_pos = delua[INDEX4(d,gx,gy+1,gz)];
      delua_neg = delua[INDEX4(d,gx,gy-1,gz)];
#endif
      delu_n[0][3*d+1] = (delu_pos - delu_neg)*dely;
      delu_n[1][3*d+1] = (fabsf(delu_pos) + fabsf(delu_neg))*dely;
      delu_n[2][3*d+1] = (delua_pos + delua_neg)*dely;
#ifdef GPU_DEBUG
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,3*d+1)] = delu_n[0][3*d+1];
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,9+3*d+1)] = delu_n[1][3*d+1];
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,18+3*d+1)] = delu_n[2][3*d+1];
#endif

#if USE_SHARED_MEM
      delu_pos = (tz < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx][ty][tz+1]) : (delu[INDEX4(d,gx,gy,gz+1)]);
      delu_neg = (tz > 0) ? (delu_s[d][tx][ty][tz-1]) : (delu[INDEX4(d,gx,gy,gz-1)]);
      delua_pos = (tz < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx][ty][tz+1]) : (delua[INDEX4(d,gx,gy,gz+1)]);
      delua_neg = (tz > 0) ? (delua_s[d][tx][ty][tz-1]) : (delua[INDEX4(d,gx,gy,gz-1)]);
#else
      delu_pos = delu[INDEX4(d,gx,gy,gz+1)];
      delu_neg = delu[INDEX4(d,gx,gy,gz-1)];
      delua_pos = delua[INDEX4(d,gx,gy,gz+1)];
      delua_neg = delua[INDEX4(d,gx,gy,gz-1)];
#endif
      delu_n[0][3*d+2] = (delu_pos - delu_neg)*delz;
      delu_n[1][3*d+2] = (fabsf(delu_pos) + fabsf(delu_neg))*delz;
      delu_n[2][3*d+2] = (delua_pos + delua_neg)*delz;
#ifdef GPU_DEBUG
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,3*d+2)] = delu_n[0][3*d+2];
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,9+3*d+2)] = delu_n[1][3*d+2];
      delu_n_g[INDEX4C(gx-2,gy-2,gz-2,18+3*d+2)] = delu_n[2][3*d+2];
#endif
    }

    for (int dd = 0; dd < NUM_DIMS * NUM_DIMS; dd++) {
      num = num + pow(delu_n[0][dd], 2.);
      denom = denom + pow(delu_n[1][dd], 2.) + (refine_filter * delu_n[2][dd]) * 2;
    }

    if (denom == 0. && num != 0.) {
      error = FLT_MAX;
    }
    else if (denom != 0.0) {
      error = num/denom;
    }

    // store error in global memory
    errors[ERR_INDEX(gx,gy,gz)] = error;
  }
#undef INDEX4
#ifdef GPU_DEBUG
#undef INDEX4C
#endif
#undef ERR_INDEX
}

#ifdef GPU_DEBUG
float invokeDecisionKernel(float *u, float *errors, float *delu_n, float refine_filter, float dx, float dy, float dz, int block_size) {
#else
float invokeDecisionKernel(float *u, float refine_filter, float dx, float dy, float dz, int block_size) {
#endif
  float error; // maximum error value to be returned

#if USE_GPUMANAGER
  /***** With GPUManager *****/

#else
  /***** Without GPUManager *****/

  // pinned host memory allocations
  float *h_error;
  gpuSafe(cudaMallocHost(&h_error, sizeof(float)));
  size_t errors_size = sizeof(float)*(block_size-2)*(block_size-2)*(block_size-2);
#ifdef GPU_DEBUG
  float *h_delu_n;
  size_t delu_n_size = sizeof(float)*(block_size-2)*(block_size-2)*(block_size-2)*3*NUM_DIMS*NUM_DIMS;
  gpuSafe(cudaMallocHost(&h_delu_n, delu_n_size));
  float *h_errors;
  gpuSafe(cudaMallocHost(&h_errors, errors_size));
#endif

  // create stream
  cudaStream_t decisionStream;
  gpuSafe(cudaStreamCreate(&decisionStream));

  // allocate device memory
  float *d_error, *d_errors;
  float *d_u, *d_delu, *d_delua;
  size_t u_size = sizeof(float)*(block_size+2)*(block_size+2)*(block_size+2);
  size_t delu_size = NUM_DIMS * u_size;
  gpuSafe(cudaMalloc(&d_u, u_size));
  gpuSafe(cudaMalloc(&d_delu, delu_size));
  gpuSafe(cudaMalloc(&d_delua, delu_size));
  gpuSafe(cudaMalloc(&d_error, sizeof(float)));
  gpuSafe(cudaMemset(d_error, 0, sizeof(float)));
  gpuSafe(cudaMalloc(&d_errors, errors_size));
#ifdef GPU_DEBUG
  float *d_delu_n;
  gpuSafe(cudaMalloc(&d_delu_n, delu_n_size));
#endif

  // copy u to device
  gpuSafe(cudaMemcpyAsync(d_u, u, u_size, cudaMemcpyHostToDevice, decisionStream));

  // execute first kernel to calculate delu and delua
  int sub_block_cnt = ceil((float)(block_size+2)/SUB_BLOCK_SIZE);
  dim3 dimGrid(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  dim3 dimBlock(SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE);
  decisionKernel1<<<dimGrid, dimBlock, 0, decisionStream>>>(d_u, d_delu, d_delua, dx, dy, dz, block_size);
  gpuCheck();

  // execute second kernel to calculate errors
  sub_block_cnt = ceil((float)block_size/SUB_BLOCK_SIZE);
  dimGrid = dim3(sub_block_cnt, sub_block_cnt, sub_block_cnt);
#ifdef GPU_DEBUG
  decisionKernel2<<<dimGrid, dimBlock, 0, decisionStream>>>(d_delu, d_delua, d_delu_n, d_errors, refine_filter, dx, dy, dz, block_size);
#else
  decisionKernel2<<<dimGrid, dimBlock, 0, decisionStream>>>(d_delu, d_delua, d_errors, refine_filter, dx, dy, dz, block_size);
#endif
  gpuCheck();

#ifdef GPU_DEBUG
  // copy data to be compared from the device
  gpuSafe(cudaMemcpyAsync(h_delu_n, d_delu_n, delu_n_size, cudaMemcpyDeviceToHost, decisionStream));
  gpuSafe(cudaMemcpyAsync(h_errors, d_errors, errors_size, cudaMemcpyDeviceToHost, decisionStream));
#endif

  // wait until completion
  gpuSafe(cudaStreamSynchronize(decisionStream));

#ifdef GPU_DEBUG
  // memcpy to regular host memory
  memcpy(delu_n, h_delu_n, delu_n_size);
  memcpy(errors, h_errors, errors_size);
#endif

  // max reduction using cub (can multiple instances of this run concurrently?)
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_errors, d_error, (block_size-2)*(block_size-2)*(block_size-2));
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_errors, d_error, (block_size-2)*(block_size-2)*(block_size-2));
  gpuSafe(cudaMemcpyAsync(h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost, decisionStream));

  // wait until completion
  gpuSafe(cudaDeviceSynchronize());

  // store max error
  error = *h_error;

  // free memory allocations
  gpuSafe(cudaFree(d_u));
  gpuSafe(cudaFree(d_delu));
  gpuSafe(cudaFree(d_delua));
  gpuSafe(cudaFree(d_error));
  gpuSafe(cudaFree(d_errors));
  gpuSafe(cudaFree(d_temp_storage));
  gpuSafe(cudaFreeHost(h_error));
#ifdef GPU_DEBUG
  gpuSafe(cudaFree(d_delu_n));
  gpuSafe(cudaFreeHost(h_delu_n));
  gpuSafe(cudaFreeHost(h_errors));
#endif

  // destroy stream
  gpuSafe(cudaStreamDestroy(decisionStream));
#endif // USE_GPUMANAGER

  return error;
}
