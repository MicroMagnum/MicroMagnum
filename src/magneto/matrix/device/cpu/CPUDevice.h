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

#ifndef CPU_DEVICE_H
#define CPU_DEVICE_H

#include "config.h"

#include "../Device.h"
#include "CPUArray.h"

namespace matty {

class Array;

class CPUDevice : public Device
{
public:
	CPUDevice();
	virtual ~CPUDevice();

	virtual CPUArray *makeArray(const Shape &shape);
	virtual void destroyArray(Array *arr);

	virtual void copyFromMemory(Array *dst, const CPUArray *src);
	virtual void copyToMemory(CPUArray *dst, const Array *src);

	virtual void slice(Array *dst_, int dst_x0, int dst_x1, const Array *src_, int src_x0, int src_x1);

	virtual void clear(Array *A);
	virtual void fill(Array *A, double value);
	virtual void assign(Array *A, const Array *op);
	virtual void add(Array *A, const Array *B, double scale);
	virtual void multiply(Array *A, const Array *B);
	virtual void divide(Array *A, const Array *B);
	virtual void scale(Array *A, double factor);
	virtual void randomize(Array *A);

	virtual void normalize3(Array *x0, Array *x1, Array *x2, double len);
	virtual void normalize3(Array *x0, Array *x1, Array *x2, const Array *len);
	virtual double absmax3(const Array *x0, const Array *x1, const Array *x2);
	virtual double sumdot3(const Array *x0, const Array *x1, const Array *x2, 
                               const Array *y0, const Array *y1, const Array *y2);

	virtual double minimum(const Array *A);
	virtual double maximum(const Array *A);
	virtual double sum(const Array *A);
	virtual double average(const Array *A);
	virtual double dot(const Array *op1, const Array *op2);
};

} // ns

#endif
