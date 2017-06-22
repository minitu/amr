#include <cub/cub.cuh>
#include <math_constants.h>
#include <math_functions.h>
#include <stdio.h>
#include <cfloat>

#if defined(USE_GPUMANAGER) || defined(USE_OLD_GPUMANAGER)
#include "wr.h"
#endif

#define USE_CUB 0
#define USE_SHARED_MEM 1
#define SUB_BLOCK_SIZE 8
#define NUM_DIMS 3

#define gpuSafe(retval) gpuPrintErr((retval), __FILE__, __LINE__)
#define gpuCheck() gpuPrintErr(cudaGetLastError(), __FILE__, __LINE__)

inline void gpuPrintErr(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess)
    fprintf(stderr,"CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
}

void mem_allocate_host(void** ptr, size_t size) {
  gpuSafe(cudaMallocHost(ptr, size));
}

void mem_deallocate_host(void* ptr) {
  gpuSafe(cudaFreeHost(ptr));
}

__device__ static float atomicMax(float* address, float val)
{
  int* address_as_i = (int*) address;
  int old = *address_as_i, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_i, assumed,
        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
  } while (assumed != old);
  return __int_as_float(old);
}

__global__ void computeKernel(float* u2, float* u, float dx, float dy, float dz, float dt, float apx, float apy, float apz, float anx, float any, float anz, int block_size) {
#define MY_INDEX(i,j,k) (((k) * (block_size+2) + (j)) * (block_size+2) + (i))
  float up[3];
  float un[3];

  int gx = blockDim.x * blockIdx.x + threadIdx.x;
  int gy = blockDim.y * blockIdx.y + threadIdx.y;
  int gz = blockDim.z * blockIdx.z + threadIdx.z;

#if USE_SHARED_MEM
  __shared__ float u_s[SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // read u into shared memory
  if ((gx < (block_size + 2)) && (gy < (block_size + 2)) && (gz < (block_size + 2))) {
    u_s[tx][ty][tz] = u[MY_INDEX(gx,gy,gz)];
  }
  __syncthreads();
#endif

  if (((gx >= 1 && gx <= block_size) && (gy >= 1 && gy <= block_size)) && (gz >= 1 && gz <= block_size)) {
#if USE_SHARED_MEM
    up[0] = (((tx < SUB_BLOCK_SIZE-1) ? (u_s[tx+1][ty][tz]) : (u[MY_INDEX(gx+1,gy,gz)])) - u[MY_INDEX(gx,gy,gz)])/dx;
    un[0] = (u[MY_INDEX(gx,gy,gz)] - ((tx > 0) ? (u_s[tx-1][ty][tz]) : (u[MY_INDEX(gx-1,gy,gz)])))/dx;
    up[1] = (((ty < SUB_BLOCK_SIZE-1) ? (u_s[tx][ty+1][tz]) : (u[MY_INDEX(gx,gy+1,gz)])) - u[MY_INDEX(gx,gy,gz)])/dy;
    un[1] = (u[MY_INDEX(gx,gy,gz)] - ((ty > 0) ? (u_s[tx][ty-1][tz]) : (u[MY_INDEX(gx,gy-1,gz)])))/dy;
    up[2] = (((tz < SUB_BLOCK_SIZE-1) ? (u_s[tx][ty][tz+1]) : (u[MY_INDEX(gx,gy,gz+1)])) - u[MY_INDEX(gx,gy,gz)])/dz;
    un[2] = (u[MY_INDEX(gx,gy,gz)] - ((tz > 0) ? (u_s[tx][ty][tz-1]) : (u[MY_INDEX(gx,gy,gz-1)])))/dz;
#else
    up[0] = (u[MY_INDEX(gx+1,gy,gz)] - u[MY_INDEX(gx,gy,gz)])/dx;
    un[0] = (u[MY_INDEX(gx,gy,gz)] - u[MY_INDEX(gx-1,gy,gz)])/dx;
    up[1] = (u[MY_INDEX(gx,gy+1,gz)] - u[MY_INDEX(gx,gy,gz)])/dy;
    un[1] = (u[MY_INDEX(gx,gy,gz)] - u[MY_INDEX(gx,gy-1,gz)])/dy;
    up[2] = (u[MY_INDEX(gx,gy,gz+1)] - u[MY_INDEX(gx,gy,gz)])/dz;
    un[2] = (u[MY_INDEX(gx,gy,gz)] - u[MY_INDEX(gx,gy,gz-1)])/dz;
#endif

    u2[MY_INDEX(gx,gy,gz)] = u[MY_INDEX(gx,gy,gz)] - dt*(apx*un[0] + anx*up[0]) - dt*(apy*un[1] + any*up[1]) - dt*(apz*un[2] + anz*up[2]);
  }
#undef MY_INDEX
}

__global__ void computeAddKernel(float* u, float* u2, float* u3, int block_size) {
#define MY_INDEX(i,j,k) (((k) * (block_size+2) + (j)) * (block_size+2) + (i))
  int gx = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int gy = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int gz = blockDim.z * blockIdx.z + threadIdx.z + 1;

  u[MY_INDEX(gx,gy,gz)] = 0.5*(u2[MY_INDEX(gx,gy,gz)] + u3[MY_INDEX(gx,gy,gz)]);
#undef MY_INDEX
}

void invokeComputeKernel(cudaStream_t computeStream, float* u, float* d_u, float* d_u2, float* d_u3, float dx, float dy, float dz, float dt, float apx, float apy, float apz, float anx, float any, float anz, int block_size, void* cb) {
#ifdef USE_GPUMANAGER
  computeStream = hapi_getStream();
#endif

  // copy u to device
  size_t u_size = sizeof(float)*(block_size+2)*(block_size+2)*(block_size+2);
  gpuSafe(cudaMemcpyAsync(d_u, u, u_size, cudaMemcpyHostToDevice, computeStream));
  gpuSafe(cudaMemcpyAsync(d_u2, u, u_size, cudaMemcpyHostToDevice, computeStream));
  gpuSafe(cudaMemcpyAsync(d_u3, u, u_size, cudaMemcpyHostToDevice, computeStream));

  // execute first kernel to calculate u2
  int sub_block_cnt = ceil((float)(block_size+2)/SUB_BLOCK_SIZE);
  dim3 dimGrid(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  dim3 dimBlock(SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE);
  computeKernel<<<dimGrid, dimBlock, 0, computeStream>>>(d_u2, d_u, dx, dy, dz, dt, apx, apy, apz, anx, any, anz, block_size);
  gpuCheck();

  // execute second kernel to calculate u3
  computeKernel<<<dimGrid, dimBlock, 0, computeStream>>>(d_u3, d_u2, dx, dy, dz, dt, apx, apy, apz, anx, any, anz, block_size);
  gpuCheck();

  // execute last kernel to calculate new u
  sub_block_cnt = ceil((float)(block_size)/SUB_BLOCK_SIZE);
  dimGrid = dim3(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  dimBlock = dim3(SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE);
  computeAddKernel<<<dimGrid, dimBlock, 0, computeStream>>>(d_u, d_u2, d_u3, block_size);
  gpuCheck();

  // copy new u to host
  gpuSafe(cudaMemcpyAsync(u, d_u, u_size, cudaMemcpyDeviceToHost, computeStream));

#ifdef USE_GPUMANAGER
  // insert callback
  hapi_insertCallback(computeStream, cb, NULL, 0);
#else // #ifdef USE_GPU
  // wait until completion
  gpuSafe(cudaStreamSynchronize(computeStream));
#endif
}

__global__ void decisionKernel1(float *u, float *delu, float *delua, float dx, float dy, float dz, int block_size) {
#define MY_INDEX(i,j,k) (((k) * (block_size+2) + (j)) * (block_size+2) + (i))
#define MY_INDEX4(d,i,j,k) ((((d) * (block_size+2) + (k)) * (block_size+2) + (j)) * (block_size+2) + (i))
  float delx = 0.5/dx;
  float dely = 0.5/dy;
  float delz = 0.5/dz;

  int gx = blockDim.x * blockIdx.x + threadIdx.x;
  int gy = blockDim.y * blockIdx.y + threadIdx.y;
  int gz = blockDim.z * blockIdx.z + threadIdx.z;

#if USE_SHARED_MEM
  __shared__ float u_s[SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // read u into shared memory
  if ((gx < (block_size + 2)) && (gy < (block_size + 2)) && (gz < (block_size + 2))) {
    u_s[tx][ty][tz] = u[MY_INDEX(gx,gy,gz)];
  }
  __syncthreads();
#endif

  // calculate differentials
  float u_pos, u_neg;
  if (((gx >= 1 && gx <= block_size) && (gy >= 1 && gy <= block_size)) && (gz >= 1 && gz <= block_size)) {
    // d/dx
#if USE_SHARED_MEM
    u_pos = (tx < SUB_BLOCK_SIZE-1) ? (u_s[tx+1][ty][tz]) : (u[MY_INDEX(gx+1,gy,gz)]);
    u_neg = (tx > 0) ? (u_s[tx-1][ty][tz]) : (u[MY_INDEX(gx-1,gy,gz)]);
#else
    u_pos = u[MY_INDEX(gx+1,gy,gz)];
    u_neg = u[MY_INDEX(gx-1,gy,gz)];
#endif
    delu[MY_INDEX4(0,gx,gy,gz)] = (u_pos - u_neg)*delx;
    delua[MY_INDEX4(0,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*delx;

    // d/dy
#if USE_SHARED_MEM
    u_pos = (ty < SUB_BLOCK_SIZE-1) ? (u_s[tx][ty+1][tz]) : (u[MY_INDEX(gx,gy+1,gz)]);
    u_neg = (ty > 0) ? (u_s[tx][ty-1][tz]) : (u[MY_INDEX(gx,gy-1,gz)]);
#else
    u_pos = u[MY_INDEX(gx,gy+1,gz)];
    u_neg = u[MY_INDEX(gx,gy-1,gz)];
#endif
    delu[MY_INDEX4(1,gx,gy,gz)] = (u_pos - u_neg)*dely;
    delua[MY_INDEX4(1,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*dely;

    // d/dz
#if USE_SHARED_MEM
    u_pos = (tz < SUB_BLOCK_SIZE-1) ? (u_s[tx][ty][tz+1]) : (u[MY_INDEX(gx,gy,gz+1)]);
    u_neg = (tz > 0) ? (u_s[tx][ty][tz-1]) : (u[MY_INDEX(gx,gy,gz-1)]);
#else
    u_pos = u[MY_INDEX(gx,gy,gz+1)];
    u_neg = u[MY_INDEX(gx,gy,gz-1)];
#endif
    delu[MY_INDEX4(2,gx,gy,gz)] = (u_pos - u_neg)*delz;
    delua[MY_INDEX4(2,gx,gy,gz)] = (fabsf(u_pos) + fabsf(u_neg))*delz;
  }
#undef MY_INDEX
#undef MY_INDEX4
}

__global__ void decisionKernel2(float *delu, float *delua, float *errors, float refine_filter, float dx, float dy, float dz, int block_size) {
#define MY_INDEX4(d,i,j,k) ((((d) * (block_size+2) + (k)) * (block_size+2) + (j)) * (block_size+2) + (i))
#define ERR_INDEX(i,j,k) ((((k)-2) * (block_size-2) + ((j)-2)) * (block_size-2) + ((i)-2))
  float delx = 0.5/dx;
  float dely = 0.5/dy;
  float delz = 0.5/dz;
  float delu_n[3][NUM_DIMS * NUM_DIMS];

  int gx = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int gy = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int gz = blockDim.z * blockIdx.z + threadIdx.z + 1;

#if USE_SHARED_MEM
  __shared__ float delu_s[NUM_DIMS][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];
  __shared__ float delua_s[NUM_DIMS][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE][SUB_BLOCK_SIZE];
#if !USE_CUB
  __shared__ float maxError;
#endif // USE_CUB

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  // read delu & delua into shared memory
  if (gx <= block_size && gy <= block_size && gz <= block_size) {
    for (int d = 0; d < NUM_DIMS; d++) {
      delu_s[d][tx][ty][tz] = delu[MY_INDEX4(d,gx,gy,gz)];
      delua_s[d][tx][ty][tz] = delua[MY_INDEX4(d,gx,gy,gz)];
    }
  }
#if !USE_CUB
  if (tx == 0 && ty == 0 && tz == 0)
    maxError = 0;
#endif // USE_CUB
  __syncthreads();
#endif // USE_SHARED_MEM

  // calculate error per thread
  float delu_pos, delu_neg;
  float delua_pos, delua_neg;
  float num = 0., denom = 0.;
  float error = 0.0f;
  if ((gx > 1 && gx < block_size) && (gy > 1 && gy < block_size) && (gz > 1 && gz < block_size)) {
    for (int d = 0; d < NUM_DIMS; d++) {
#if USE_SHARED_MEM
      delu_pos = (tx < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx+1][ty][tz]) : (delu[MY_INDEX4(d,gx+1,gy,gz)]);
      delu_neg = (tx > 0) ? (delu_s[d][tx-1][ty][tz]) : (delu[MY_INDEX4(d,gx-1,gy,gz)]);
      delua_pos = (tx < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx+1][ty][tz]) : (delua[MY_INDEX4(d,gx+1,gy,gz)]);
      delua_neg = (tx > 0) ? (delua_s[d][tx-1][ty][tz]) : (delua[MY_INDEX4(d,gx-1,gy,gz)]);
#else
      delu_pos = delu[MY_INDEX4(d,gx+1,gy,gz)];
      delu_neg = delu[MY_INDEX4(d,gx-1,gy,gz)];
      delua_pos = delua[MY_INDEX4(d,gx+1,gy,gz)];
      delua_neg = delua[MY_INDEX4(d,gx-1,gy,gz)];
#endif
      delu_n[0][3*d+0] = (delu_pos - delu_neg)*delx;
      delu_n[1][3*d+0] = (fabsf(delu_pos) + fabsf(delu_neg))*delx;
      delu_n[2][3*d+0] = (delua_pos + delua_neg)*delx;

#if USE_SHARED_MEM
      delu_pos = (ty < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx][ty+1][tz]) : (delu[MY_INDEX4(d,gx,gy+1,gz)]);
      delu_neg = (ty > 0) ? (delu_s[d][tx][ty-1][tz]) : (delu[MY_INDEX4(d,gx,gy-1,gz)]);
      delua_pos = (ty < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx][ty+1][tz]) : (delua[MY_INDEX4(d,gx,gy+1,gz)]);
      delua_neg = (ty > 0) ? (delua_s[d][tx][ty-1][tz]) : (delua[MY_INDEX4(d,gx,gy-1,gz)]);
#else
      delu_pos = delu[MY_INDEX4(d,gx,gy+1,gz)];
      delu_neg = delu[MY_INDEX4(d,gx,gy-1,gz)];
      delua_pos = delua[MY_INDEX4(d,gx,gy+1,gz)];
      delua_neg = delua[MY_INDEX4(d,gx,gy-1,gz)];
#endif
      delu_n[0][3*d+1] = (delu_pos - delu_neg)*dely;
      delu_n[1][3*d+1] = (fabsf(delu_pos) + fabsf(delu_neg))*dely;
      delu_n[2][3*d+1] = (delua_pos + delua_neg)*dely;

#if USE_SHARED_MEM
      delu_pos = (tz < SUB_BLOCK_SIZE-1) ? (delu_s[d][tx][ty][tz+1]) : (delu[MY_INDEX4(d,gx,gy,gz+1)]);
      delu_neg = (tz > 0) ? (delu_s[d][tx][ty][tz-1]) : (delu[MY_INDEX4(d,gx,gy,gz-1)]);
      delua_pos = (tz < SUB_BLOCK_SIZE-1) ? (delua_s[d][tx][ty][tz+1]) : (delua[MY_INDEX4(d,gx,gy,gz+1)]);
      delua_neg = (tz > 0) ? (delua_s[d][tx][ty][tz-1]) : (delua[MY_INDEX4(d,gx,gy,gz-1)]);
#else
      delu_pos = delu[MY_INDEX4(d,gx,gy,gz+1)];
      delu_neg = delu[MY_INDEX4(d,gx,gy,gz-1)];
      delua_pos = delua[MY_INDEX4(d,gx,gy,gz+1)];
      delua_neg = delua[MY_INDEX4(d,gx,gy,gz-1)];
#endif
      delu_n[0][3*d+2] = (delu_pos - delu_neg)*delz;
      delu_n[1][3*d+2] = (fabsf(delu_pos) + fabsf(delu_neg))*delz;
      delu_n[2][3*d+2] = (delua_pos + delua_neg)*delz;
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

#if USE_CUB
    // store error in global memory
    errors[ERR_INDEX(gx,gy,gz)] = error;
#else
#if USE_SHARED_MEM
    atomicMax(&maxError, error);
#else
    atomicMax(errors, error);
#endif // USE_SHARED_MEM
#endif // USE_CUB
  }

#if !USE_CUB
#if USE_SHARED_MEM
  __syncthreads();
  if (tx == 0 && ty == 0 && tz == 0)
    atomicMax(errors, maxError);
#endif // USE_SHARED_MEM
#endif // USE_CUB
#undef MY_INDEX4
#undef ERR_INDEX
}

#if defined(USE_GPU) || defined(USE_GPUMANAGER)
#ifdef USE_GPU
void createStream(cudaStream_t* stream_ptr) {
  gpuSafe(cudaStreamCreate(stream_ptr));
}

void destroyStream(cudaStream_t stream) {
  gpuSafe(cudaStreamDestroy(stream));
}
#endif

void allocateHostMemory(void** ptr, size_t size) {
  gpuSafe(cudaMallocHost(ptr, size));
}

void freeHostMemory(void* ptr) {
  gpuSafe(cudaFreeHost(ptr));
}

void allocateDeviceMemory(void** ptr, size_t size) {
  gpuSafe(cudaMalloc(ptr, size));
}

void freeDeviceMemory(void* ptr) {
  gpuSafe(cudaFree(ptr));
}

float invokeDecisionKernel(cudaStream_t decisionStream, float* u, float* h_error, float* d_error, float* d_u, float* d_delu, float* d_delua, float refine_filter, float dx, float dy, float dz, int block_size, void* cb) {
#ifdef USE_GPU
  float error = 0.0; // maximum error value to be returned
#endif

#if USE_CUB
  size_t errors_size = sizeof(float)*(block_size-2)*(block_size-2)*(block_size-2);
  float *d_errors;
  gpuSafe(cudaMalloc(&d_errors, errors_size));
#endif

#ifdef USE_GPUMANAGER
  decisionStream = hapi_getStream();
#endif

  // memset
  size_t u_size = sizeof(float)*(block_size+2)*(block_size+2)*(block_size+2);
  size_t delu_size = NUM_DIMS * u_size;
  *h_error = 0.0f;
  gpuSafe(cudaMemset(d_delu, 0, delu_size));
  gpuSafe(cudaMemset(d_delua, 0, delu_size));
  gpuSafe(cudaMemset(d_error, 0, sizeof(float)));

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
#if USE_CUB
  decisionKernel2<<<dimGrid, dimBlock, 0, decisionStream>>>(d_delu, d_delua, d_errors, refine_filter, dx, dy, dz, block_size);
#else
  decisionKernel2<<<dimGrid, dimBlock, 0, decisionStream>>>(d_delu, d_delua, d_error, refine_filter, dx, dy, dz, block_size);
#endif
  gpuSafe(cudaMemcpyAsync(h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost, decisionStream));
  gpuCheck();

#ifdef USE_GPU
  // wait until completion
  gpuSafe(cudaStreamSynchronize(decisionStream));
#else // #ifdef USE_GPUMANAGER
  // insert callback
  hapi_insertCallback(decisionStream, cb, NULL, 0);
#endif

#if USE_CUB
  // max reduction using cub (can multiple instances of this run concurrently?)
  void *d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_errors, d_error, (block_size-2)*(block_size-2)*(block_size-2));
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_errors, d_error, (block_size-2)*(block_size-2)*(block_size-2));
  gpuSafe(cudaMemcpyAsync(h_error, d_error, sizeof(float), cudaMemcpyDeviceToHost, decisionStream));

  // wait until completion
  gpuSafe(cudaDeviceSynchronize());

  // free memory allocations
  gpuSafe(cudaFree(d_errors));
  gpuSafe(cudaFree(d_temp_storage));
#endif

#ifdef USE_GPU
  // store max error
  error = *h_error;

  return error;
#else // #ifdef USE_GPUMANAGER
  return 0;
#endif
}
#else // #ifdef USE_OLD_GPUMANAGER
void run_DECISION_KERNEL_1(workRequest *wr, cudaStream_t kernel_stream, void **devBuffers) {
  void* constants = wr->getUserData();
  float dx = *((float*)constants);
  float dy = *((float*)((char*)constants + sizeof(float)));
  float dz = *((float*)((char*)constants + sizeof(float)*2));
  int block_size = *((int*)((char*)constants + sizeof(float)*3));
  decisionKernel1<<<wr->dimGrid, wr->dimBlock, wr->smemSize, kernel_stream>>>
    ((float*)hapi_devBuffer(wr, 0), (float*)hapi_devBuffer(wr, 1), (float*)hapi_devBuffer(wr, 2),
     dx, dy, dz, block_size);
  gpuCheck();
}

void run_DECISION_KERNEL_2(workRequest *wr, cudaStream_t kernel_stream, void **devBuffers) {
  void* constants = wr->getUserData();
  float dx = *((float*)constants);
  float dy = *((float*)((char*)constants + sizeof(float)));
  float dz = *((float*)((char*)constants + sizeof(float)*2));
  int block_size = *((int*)((char*)constants + sizeof(float)*3));
  float refine_filter = *((float*)((char*)constants + sizeof(float)*3 + sizeof(int)));
  decisionKernel2<<<wr->dimGrid, wr->dimBlock, wr->smemSize, kernel_stream>>>
    ((float*)hapi_devBuffer(wr, 0), (float*)hapi_devBuffer(wr, 1), (float*)hapi_devBuffer(wr, 2),
     refine_filter, dx, dy, dz, block_size);
  gpuCheck();
}

void* getPinnedMemory(size_t size) {
  return hapi_poolMalloc(size);
}

void freePinnedMemory(void* ptr) {
  hapi_poolFree(ptr);
}

int invokeDecisionKernel(float* u, float* pinned_error, float refine_filter, float dx, float dy, float dz, int block_size, int chare_index, int streamID, void* cb) {
  // sizes of data
  size_t u_size = sizeof(float)*(block_size+2)*(block_size+2)*(block_size+2);
  size_t delu_size = NUM_DIMS * u_size;

  // store constants
  void* constants = malloc(sizeof(float)*3 + sizeof(int) + sizeof(float));
  *((float*)constants) = dx;
  *((float*)((char*)constants + sizeof(float))) = dy;
  *((float*)((char*)constants + sizeof(float)*2)) = dz;
  *((int*)((char*)constants + sizeof(float)*3)) = block_size;
  *((float*)((char*)constants + sizeof(float)*3 + sizeof(int))) = refine_filter;

  // create work request for first kernel
  workRequest* decision1 = hapi_createWorkRequest();
  int sub_block_cnt = ceil((float)(block_size+2)/SUB_BLOCK_SIZE);
  dim3 dimGrid(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  dim3 dimBlock(SUB_BLOCK_SIZE, SUB_BLOCK_SIZE, SUB_BLOCK_SIZE);
  decision1->setExecParams(dimGrid, dimBlock);
  if (streamID != -1)
    decision1->setStream(streamID);
  decision1->addBufferInfo(u, u_size, true, false, false, chare_index*4); // d_u
  decision1->addBufferInfo(NULL, delu_size, false, false, false, chare_index*4 + 1); // d_delu
  decision1->addBufferInfo(NULL, delu_size, false, false, false, chare_index*4 + 2); // d_delua
  decision1->setUserData(constants, sizeof(float)*3 + sizeof(int));
  decision1->setRunKernel(run_DECISION_KERNEL_1);

  // enqueue first work request
  streamID = hapi_enqueue(decision1);

  // create work request for second kernel
  workRequest* decision2 = hapi_createWorkRequest();
  sub_block_cnt = ceil((float)block_size/SUB_BLOCK_SIZE);
  dimGrid = dim3(sub_block_cnt, sub_block_cnt, sub_block_cnt);
  decision2->setExecParams(dimGrid, dimBlock);
  decision2->setStream(streamID);
  //decision2->addBufferInfo(NULL, u_size, false, false, true, chare_index*4); // d_u
  decision2->addBufferInfo(NULL, delu_size, false, false, false, chare_index*4 + 1); // d_delu
  decision2->addBufferInfo(NULL, delu_size, false, false, false, chare_index*4 + 2); // d_delua
  decision2->addBufferInfo(pinned_error, sizeof(float), true, true, false, chare_index*4 + 3); // d_error
  decision2->setCallback(cb);
  decision2->setUserData(constants, sizeof(float)*3 + sizeof(int) + sizeof(float));
  decision2->setRunKernel(run_DECISION_KERNEL_2);

  // enqueue second work request
  hapi_enqueue(decision2);

  return streamID;
}
#endif
