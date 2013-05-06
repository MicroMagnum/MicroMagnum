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

#ifndef CUDA_TOOLS_H
#define CUDA_TOOLS_H

#include "config.h"

#include <driver_types.h>
#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
