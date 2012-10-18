#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include "config.h"

#include <driver_types.h>
#include <cufft.h>

void printCudaSpecs();
int getMaxGflopsDeviceId();
void initialize_cuda(int force_device = -1);
void deinitialize_cuda();

void doCheckCudaSuccess(cudaError err, const char *file, int line_no);
void doCheckCufftSuccess(cufftResult err, const char *file, int line_no);

#define checkCudaSuccess(err) doCheckCudaSuccess(err, __FILE__, __LINE__)
#define checkCufftSuccess(err) doCheckCufftSuccess(err, __FILE__, __LINE__)

void doCheckCudaLastError(const char *file, int line_no, const char *msg);
#define checkCudaLastError(msg) doCheckCudaLastError(__FILE__, __LINE__, msg)

#ifdef ENABLE_CUDA_THREAD_SYNCHRONIZE
#define CUDA_THREAD_SYNCHRONIZE() { checkCudaSuccess(cudaThreadSynchronize()); }
#else
#define CUDA_THREAD_SYNCHRONIZE() { }
#endif

#endif
