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

#ifndef DEVICE_H
#define DEVICE_H

#include "config.h"
#include <vector>
#include <ostream>

namespace matty {

class Array;
class CPUArray;
class Shape;

class Device
{
public:
	Device(const std::string &dev_name);
	virtual ~Device();

	virtual Array *makeArray(const Shape &shape) = 0;
	virtual void destroyArray(Array *arr) = 0;

	virtual void copyFromMemory(Array *dst, const CPUArray *src) = 0;
	virtual void copyToMemory(CPUArray *dst, const Array *src) = 0;

	virtual void slice(Array *dst, int dst_x0, int dst_x1, const Array *src, int src_x0, int src_x1) = 0;

	virtual void clear(Array *A) = 0;
	virtual void fill(Array *A, double value) = 0;
	virtual void assign(Array *A, const Array *op) = 0;
	virtual void add(Array *A, const Array *B, double scale) = 0;
	virtual void multiply(Array *A, const Array *B) = 0;
	virtual void divide(Array *A, const Array *B) = 0;
	virtual void scale(Array *A, double factor) = 0;
	virtual void randomize(Array *A) = 0;

	virtual void normalize3(Array *x0, Array *x1, Array *x2, double len) = 0;
	virtual void normalize3(Array *x0, Array *x1, Array *x2, const Array *len) = 0;
	virtual double absmax3(const Array *x0, const Array *x1, const Array *x2) = 0;
	virtual double sumdot3(const Array *x0, const Array *x1, const Array *x2, 
                               const Array *y0, const Array *y1, const Array *y2) = 0;

	virtual double minimum(const Array *A) = 0;
	virtual double maximum(const Array *A) = 0;
	virtual double sum(const Array *A) = 0;
	virtual double average(const Array *A) = 0;
	virtual double dot(const Array *op1, const Array *op2) = 0;

	virtual void printReport(std::ostream &out);

protected:
	std::string dev_name; // e.g. "cpu", "cuda0", "cuda1"
	int alloced_mem, alloced_arrays;
};

} // ns

#endif
