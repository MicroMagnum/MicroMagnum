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
#include "CUDADevice.h"
#include "CUDAArray.h"

#include <cassert>
#include <stdexcept>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>

#include "cublas_wrap.h"
#include "kernels_simple.h"
#include "kernels_reduce.h"

#include "Logger.h"

namespace matty {

	CU32Device::CU32Device(int cuda_device) : Device("cuda32"), CUDADeviceMixin<CU32Array>("cuda32") 
	{
	}

	CU32Device::~CU32Device()
	{
	}

	void CU32Device::copyFromMemory(Array *DST, const CPUArray *src)
	{
		CU32Array *dst = dynamic_cast<CU32Array*>(DST);
		
		LOG_DEBUG << "Memory copy: GPU32<-CPU";

		const int N0 = dst->getShape().getNumEl();
		const int N1 = src->getShape().getNumEl();
		if (N0 != N1) throw std::domain_error("CU32Device::copyFromMemory: Need to assign array of same size.");

		// Convert double array to float array.
		const double * RESTRICT src_dbl = src->ptr();
		float        * RESTRICT src_flt = new float [N0];
		for (int i=0; i<N0; ++i) {
			src_flt[i] = static_cast<float>(src_dbl[i]);
		}

		// Convert float array to gpu
		const cudaError_t err = cudaMemcpy((void*)dst->ptr(), (const void*)src_flt, N0 * sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			const std::string err_str = std::string("CU32Device::copyFromMemory: Array copy from CPU to GPU failed. Reason: ") + cudaGetErrorString(err);
			throw std::runtime_error(err_str.c_str());
		}

		// Clean up.
		delete [] src_flt;
	}

	void CU32Device::copyToMemory(CPUArray *dst, const Array *SRC)
	{
		const CU32Array *src = dynamic_cast<const CU32Array*>(SRC);

		LOG_DEBUG << "Memory copy: GPU32->CPU";

		const int N0 = dst->getShape().getNumEl();
		const int N1 = src->getShape().getNumEl();
		if (N0 != N1) throw std::domain_error("CU32Device::copyToMemory: Need to assign array of same size.");

		assert(sizeof(double) == 2*sizeof(float));

		double      *dst_dbl = dst->ptr();
		const float *dst_flt = ((float*)dst_dbl) + N0;

		// Copy from gpu to host memory
		const cudaError_t err = cudaMemcpy((void*)dst_flt, (const void*)src->ptr(), N0 * sizeof(float), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			const std::string err_str = std::string("CU32Device::copyToMemory: Array copy from GPU to CPU failed. Reason: ") + cudaGetErrorString(err);
			throw std::runtime_error(err_str.c_str());
		}

		// Convert double to float array ("in-place"!)
		for (int i=0; i<N0; ++i) {
			*dst_dbl++ = static_cast<double>(*dst_flt++);
		}
	}

	void CU32Device::slice(Array *dst, int dst_x0, int dst_x1, const Array *src, int src_x0, int src_x1)
	{
		assert(0 && "CU32Device::slice: not implemented!");
	}

#ifdef HAVE_CUDA_64
	CU64Device::CU64Device(int cuda_device) : Device("cuda64"), CUDADeviceMixin<CU64Array>("cuda64")
	{
	}

	CU64Device::~CU64Device()
	{
	}

	void CU64Device::copyFromMemory(Array *DST, const CPUArray *src)
	{
		CU64Array *dst = dynamic_cast<CU64Array*>(DST);
		
		LOG_DEBUG << "Memory copy: GPU64<-CPU";

		const int N0 = dst->getShape().getNumEl();
		const int N1 = src->getShape().getNumEl();
		if (N0 != N1) throw std::domain_error("CU64Device::copyFromMemory: Need to assign array of same size.");

		// Convert float array to gpu
		const cudaError_t err = cudaMemcpy((void*)dst->ptr(), (const void*)src->ptr(), N0 * sizeof(double), cudaMemcpyHostToDevice);
		if (err != cudaSuccess) {
			const std::string err_str = std::string("CU64Device::copyFromMemory: Array copy from CPU to GPU failed. Reason: ") + cudaGetErrorString(err);
			throw std::runtime_error(err_str.c_str());
		}
	}

	void CU64Device::copyToMemory(CPUArray *dst, const Array *SRC)
	{
		const CU64Array *src = dynamic_cast<const CU64Array*>(SRC);

		LOG_DEBUG << "Memory copy: GPU64->CPU";

		const int N0 = dst->getShape().getNumEl();
		const int N1 = src->getShape().getNumEl();
		if (N0 != N1) throw std::domain_error("CU64Device::copyToMemory: Need to assign array of same size.");

		// Copy from gpu to host memory
		const cudaError_t err = cudaMemcpy((void*)dst->ptr(), (const void*)src->ptr(), N0 * sizeof(double), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess) {
			const std::string err_str = std::string("CU64Device::copyToMemory: Array copy from GPU to CPU failed. Reason: ") + cudaGetErrorString(err);
			throw std::runtime_error(err_str.c_str());
		}
	}

	void CU64Device::slice(Array *dst, int dst_x0, int dst_x1, const Array *src, int src_x0, int src_x1)
	{
		assert(0 && "CU64Device::slice: not implemented!");
	}
#endif

} // ns
