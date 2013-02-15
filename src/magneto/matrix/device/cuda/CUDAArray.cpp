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
#include "CUDAArray.h"
#include "CUDADevice.h"

#include <stdexcept>

#include <cublas.h>
#include <cuda_runtime.h>

namespace matty {

CU32Array::CU32Array(const Shape &shape, CU32Device *device)
	: Array(shape, device), data(0), cuda_device(device)
{
	// Note: see also 'cudaMallocPitch', 'cudaMalloc3d'
	cudaError_t err = cudaMalloc((void**)&data, getShape().getNumEl() * sizeof(float));
	if (err != cudaSuccess) throw std::runtime_error("CU32Array::allocate: Could not allocate memory on GPU.");
}

CU32Array::~CU32Array()
{
	cudaFree(data);
}

#ifdef HAVE_CUDA_64
CU64Array::CU64Array(const Shape &shape, CU64Device *device)
	: Array(shape, device), data(0), cuda_device(device)
{
	// Note: see also 'cudaMallocPitch', 'cudaMalloc3d'
	cudaError_t err = cudaMalloc((void**)&data, getShape().getNumEl() * sizeof(double));
	if (err != cudaSuccess) throw std::runtime_error("CU64Array::allocate: Could not allocate memory on GPU.");
}

CU64Array::~CU64Array()
{
	cudaFree(data);
}
#endif

} // ns
