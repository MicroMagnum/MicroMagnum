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

#ifndef CUDA_DEVICE_H
#define CUDA_DEVICE_H

#include "config.h"

#include "../Device.h"
#include "../cpu/CPUDevice.h"

#include "CUDAArray.h"
#include "CUDADeviceMixin.h"

namespace matty {

class Array;

class CU32Device : private CUDADeviceMixin<CU32Array>, virtual public Device
{
public:
	CU32Device(int cuda_device);
	virtual ~CU32Device();

	virtual void copyFromMemory(Array *dst, const CPUArray *src);
	virtual void copyToMemory(CPUArray *dst, const Array *src);
	virtual void slice(Array *dst, int dst_x0, int dst_x1, const Array *src, int src_x0, int src_x1);

	virtual CU32Array *makeArray(const Shape &shape)
	{
		CU32Array *arr = new CU32Array(shape, this);
		alloced_mem    += arr->getNumBytes();
		alloced_arrays += 1;
		return arr;
	}

	virtual void destroyArray(Array *arr)
	{
		alloced_mem    -= arr->getNumBytes();
		alloced_arrays -= 1;
		delete arr;
	}
};

#ifdef HAVE_CUDA_64
class CU64Device : private CUDADeviceMixin<CU64Array>, virtual public Device
{
public:
	CU64Device(int cuda_device);
	virtual ~CU64Device();

	virtual void copyFromMemory(Array *dst, const CPUArray *src);
	virtual void copyToMemory(CPUArray *dst, const Array *src);
	virtual void slice(Array *dst, int dst_x0, int dst_x1, const Array *src, int src_x0, int src_x1);

	virtual CU64Array *makeArray(const Shape &shape)
	{
		CU64Array *arr = new CU64Array(shape, this);
		alloced_mem    += arr->getNumBytes();
		alloced_arrays += 1;
		return arr;
	}

	virtual void destroyArray(Array *arr)
	{
		alloced_mem    -= arr->getNumBytes();
		alloced_arrays -= 1;
		delete arr;
	}
};
#endif

} // ns

#endif
