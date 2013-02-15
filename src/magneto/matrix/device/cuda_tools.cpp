/*
 * Copyright 2012, 2013 by the Micromagnum authors.
 *
 * This file is part of MicroMagnum.
 * 
 * MicroMagnum is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * MicroMagnum is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "config.h"
#include "cuda_tools.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
using namespace std;

#include "Logger.h"

static int ConvertSMVer2Cores(int major, int minor)
{
	// info taken from the cuda SDK headers
	const int version = (major << 4) + minor;
	switch (version) {
		case 0x10:
		case 0x11:
		case 0x12:
		case 0x13: return 8;
		case 0x20: return 32;
		case 0x21: return 48;
	}
	return -1;
}

static const char *CufftErrorToString(cufftResult err)
{
        switch (err) {
                case CUFFT_SUCCESS:         return "CUFFT_SUCCESS";
                case CUFFT_INVALID_PLAN:    return "CUFFT_INVALID_PLAN";
                case CUFFT_ALLOC_FAILED:    return "CUFFT_ALLOC_FAILED";
                case CUFFT_INVALID_TYPE:    return "CUFFT_INVALID_TYPE";
                case CUFFT_INVALID_VALUE:   return "CUFFT_INVALID_VALUE";
                case CUFFT_INTERNAL_ERROR:  return "CUFFT_INTERNAL_ERROR";
                case CUFFT_EXEC_FAILED:     return "CUFFT_EXEC_FAILED";
                case CUFFT_SETUP_FAILED:    return "CUFFT_SETUP_FAILED";
                //case CUFFT_SHUTDOWN_FAILED: return "CUFFT_SHUTDOWN_FAILED"; // in documentation, but not defined in headers
                case CUFFT_INVALID_SIZE:    return "CUFFT_INVALID_SIZE";
		case CUFFT_UNALIGNED_DATA:  return "CUFFT_UNALIGNED_DATA";
        }
        return "(unknown cufft error code)";
}

void printCudaSpecs()
{
	LOG_INFO << "Initializing CUDA";
	LOG_DEBUG << "  Compiled with CUDA runtime version " << CUDART_VERSION;

	int deviceCount;
	if (cudaGetDeviceCount(&deviceCount) != cudaSuccess) {
		throw std::runtime_error("cudaGetDeviceCount call failed!");
	}

	if (deviceCount == 0) {
		LOG_WARN << "There is no device supporting CUDA";
	}

	for (int dev=0; dev<deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0) {
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
				LOG_WARN << "There is no device supporting CUDA";
			else
				LOG_INFO << "There are " << deviceCount << " devices supporting CUDA";
		}
		LOG_INFO  << "Device " << dev << " \"" << deviceProp.name << "\"";

		int  driverVersion = 0; cudaDriverGetVersion(&driverVersion);
		int runtimeVersion = 0; cudaRuntimeGetVersion(&runtimeVersion);
		LOG_DEBUG << "  CUDA Driver Version:                           " << driverVersion/1000 << "." << driverVersion%100;
		LOG_DEBUG << "  CUDA Runtime Version:                          " << runtimeVersion/1000 << "." << runtimeVersion%100;
		LOG_DEBUG << "  Major revision number:                         " << deviceProp.major;
		LOG_DEBUG << "  Minor revision number:                         " << deviceProp.minor;
		LOG_DEBUG << "  Total amount of global memory:                 " << deviceProp.totalGlobalMem << " bytes";
		const int num_mp       = deviceProp.multiProcessorCount;
		const int cores_per_mp = ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
		LOG_DEBUG << "  Number of multiprocessors:                     " << num_mp;
        	LOG_DEBUG << "  Multiprocessors x Cores/MP = Cores:            " << num_mp << " (MP) x " << cores_per_mp << " (Cores/MP) = " << num_mp * cores_per_mp << " (Cores)";
		LOG_DEBUG << "  Total amount of constant memory:               " << deviceProp.totalConstMem << " bytes"; 
		LOG_DEBUG << "  Total amount of shared memory per block:       " << deviceProp.sharedMemPerBlock << " bytes";
		LOG_DEBUG << "  Total number of registers available per block: " << deviceProp.regsPerBlock;
		LOG_DEBUG << "  Warp size:                                     " << deviceProp.warpSize;
		LOG_DEBUG << "  Maximum number of threads per block:           " << deviceProp.maxThreadsPerBlock;
		LOG_DEBUG << "  Maximum sizes of each dimension of a block:    " << deviceProp.maxThreadsDim[0] << "x" << deviceProp.maxThreadsDim[1] << "x" << deviceProp.maxThreadsDim[2];
		LOG_DEBUG << "  Maximum sizes of each dimension of a grid:     " << deviceProp.maxGridSize[0] << "x" << deviceProp.maxGridSize[1] << "x" << deviceProp.maxGridSize[2];
		LOG_DEBUG << "  Maximum memory pitch:                          " << deviceProp.memPitch << " bytes";
		LOG_DEBUG << "  Texture alignment:                             " << deviceProp.textureAlignment << " bytes";
		LOG_DEBUG << "  Clock rate:                                    " << deviceProp.clockRate * 1e-6f << " GHz";
		LOG_DEBUG << "  Concurrent copy and execution:                 " << (deviceProp.deviceOverlap ? "Yes" : "No");
		LOG_DEBUG << "  Run time limit on kernels:                     " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		LOG_DEBUG << "  Integrated:                                    " << (deviceProp.integrated ? "Yes" : "No");
		LOG_DEBUG << "  Support host page-locked memory mapping:       " << (deviceProp.canMapHostMemory ? "Yes" : "No");
		const char *mode = "Unknown";
		if (deviceProp.computeMode == cudaComputeModeDefault) {
			mode = "Default (multiple host threads can use this device simultaneously)";
		} else if (deviceProp.computeMode == cudaComputeModeExclusive) {
			mode = "Exclusive (only one host thread at a time can use this device)";
		} else if (deviceProp.computeMode == cudaComputeModeProhibited) {
			mode = "Prohibited (no host thread can use this device)";
		}
		LOG_DEBUG << "  Compute mode:                                  " << mode;
		LOG_DEBUG << "  Concurrent kernel execution:                   " << (deviceProp.concurrentKernels ? "Yes" : "No");
		LOG_DEBUG << "  Device has ECC support enabled:                " << (deviceProp.ECCEnabled ? "Yes" : "No");
		LOG_DEBUG << "  Device is using TCC driver mode:               " << (deviceProp.tccDriver ? "Yes" : "No");
	}
}

int getMaxGflopsDeviceId()
{
	int device_count = 0;
	cudaGetDeviceCount(&device_count);

	cudaDeviceProp device_properties;
	int max_gflops_device = 0;
	int max_gflops = 0;
	
	int current_device = 0;
	cudaGetDeviceProperties(&device_properties, current_device);
	max_gflops = device_properties.multiProcessorCount * device_properties.clockRate;
	++current_device;

	while (current_device < device_count) {
		cudaGetDeviceProperties( &device_properties, current_device );
		int gflops = device_properties.multiProcessorCount * device_properties.clockRate;
		if (gflops > max_gflops) {
			max_gflops        = gflops;
			max_gflops_device = current_device;
		}
		++current_device;
	}

	return max_gflops_device;
}

static bool cuda_initialized = false;

void initialize_cuda(int force_device)
{
	if (cuda_initialized) return;

	printCudaSpecs();

	int device = -1;
	if (force_device == -1) {
		//device = getMaxGflopsDeviceId();
		LOG_INFO << "Auto-selecting available GPU.";
	} else {
		device = force_device;
		LOG_INFO << "Using GPU device with id " << device << " (set by user)";
		cudaSetDevice(device);
	}

	cudaSetDeviceFlags(cudaDeviceScheduleSpin);

	cuda_initialized = true;
}

void deinitialize_cuda()
{
	cudaThreadExit();
	cuda_initialized = false;
}

void doCheckCudaSuccess(cudaError err, const char *file, int line_no)
{
	if (err != cudaSuccess) {
		LOG_ERROR << "checkCudaSuccess(): Runtime error " << err << "at " << file << ":" << line_no;
		exit(-1);
	}
}

void doCheckCufftSuccess(cufftResult err, const char *file, int line_no)
{
	if (err != CUFFT_SUCCESS) {
		LOG_ERROR << "checkCufftSuccess(): Runtime error " << CufftErrorToString(err) << " at " << file << ":" << line_no;
		exit(-1);
	}
}

void doCheckCudaLastError(const char *file, int line_no, const char *msg)
{
	cudaError err = cudaGetLastError();
	if (err != cudaSuccess) {
		const char *what = cudaGetErrorString(err);
		LOG_ERROR << "checkCudaLastError(): Runtime error at " << file << ":" << line_no << " : " << msg << "; " << what;
		exit(-1);
	}
}

