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
#include "CPUDevice.h"
#include "CPUArray.h"

#include <cassert>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <stdexcept>

namespace matty {

static int cast(      Array *arr1,       CPUArray **cpu_arr1);
static int cast(const Array *arr1, const CPUArray **cpu_arr1);
static int cast(      Array *arr1, const Array *arr2,       CPUArray **cpu_arr1, const CPUArray **cpu_arr2);
static int cast(const Array *arr1, const Array *arr2, const CPUArray **cpu_arr1, const CPUArray **cpu_arr2);

CPUDevice::CPUDevice() : Device("cpu")
{
}

CPUDevice::~CPUDevice()
{
}

CPUArray *CPUDevice::makeArray(const Shape &shape)
{
	CPUArray *arr = new CPUArray(shape, this);
	alloced_mem    += arr->getNumBytes();
	alloced_arrays += 1;
	return arr;
}

void CPUDevice::destroyArray(Array *arr)
{
	alloced_mem    -= arr->getNumBytes();
	alloced_arrays -= 1;
	delete arr;
}

void CPUDevice::copyFromMemory(Array *DST, const CPUArray *src)
{
	CPUArray *dst; const int N = cast(DST, &dst);
	if (dst == src) return; // done.
	memcpy(dst->data, src->data, N * sizeof(double));
}

void CPUDevice::copyToMemory(CPUArray *dst, const Array *SRC)
{
	const CPUArray *src; const int N = cast(SRC, &src);
	if (dst == src) return; // done.
	memcpy(dst->data, src->data, N * sizeof(double));
}

void CPUDevice::slice(Array *dst_, int dst_x0, int dst_x1, const Array *src_, int src_x0, int src_x1)
{
	      CPUArray *dst = static_cast<      CPUArray*>(dst_);
	const CPUArray *src = static_cast<const CPUArray*>(src_);

	const Shape &src_shape = src->getShape();
	const int src_dim_x       = src_shape.getDim(0);
	const int src_dim_rest    = src_shape.getNumEl() / src_dim_x;
	const int src_stride_x    = 1;
	const int src_stride_rest = src_dim_x;

	const Shape &dst_shape = dst->getShape();
	const int dst_dim_x       = dst_shape.getDim(0);
	const int dst_dim_rest    = dst_shape.getNumEl() / dst_dim_x;
	const int dst_stride_x    = 1;
	const int dst_stride_rest = dst_dim_x;

	assert(src_dim_rest == dst_dim_rest);
	assert(dst_x1-dst_x0 == src_x1-src_x0);

	const int x_range = src_x1 - src_x0; // or "xrange = dst_x1 - dst_x0".

	for (int i=0; i<src_dim_rest; ++i) {
		const double *src_row = src->data + i*src_stride_rest + src_x0*src_stride_x;
		      double *dst_row = dst->data + i*dst_stride_rest + dst_x0*dst_stride_x;
		memcpy(dst_row, src_row, x_range * sizeof(double));
	}
}

void CPUDevice::clear(Array *A) // A[*] <- 0
{
	CPUArray *a; const int N = cast(A, &a);
	std::memset(a->data, 0, sizeof(double) * N);
}

void CPUDevice::fill(Array *A, double value) // A[*] <- value
{
	CPUArray *a; const int N = cast(A, &a);
	for (int i=0; i<N; ++i) a->data[i] = value;
}

void CPUDevice::assign(Array *A, const Array *B) // A[n] <- B[n]
{
	CPUArray *a; const CPUArray *b; const int N = cast(A, B, &a, &b);
	memcpy(a->data /*dst*/, b->data /*src*/, N * sizeof(double));
}

void CPUDevice::add(Array *A, const Array *B, double scale) // A[n] <- B[n] * scale
{
	CPUArray *a; const CPUArray *b; const int N = cast(A, B, &a, &b);
	if (scale == 1.0) {
		for (int i=0; i<N; ++i) {
			a->data[i] += b->data[i];
		}
	} else if (scale == -1.0) {
		for (int i=0; i<N; ++i) {
			a->data[i] -= b->data[i];
		}
	} else {
		for (int i=0; i<N; ++i) {
			a->data[i] += b->data[i] * scale;
		}
	}
}

void CPUDevice::multiply(Array *A, const Array *B)
{
	CPUArray *a; const CPUArray *b; const int N = cast(A, B, &a, &b);
	for (int i=0; i<N; ++i) {
		a->data[i] *= b->data[i];
	}
}

void CPUDevice::divide(Array *A, const Array *B)
{
	CPUArray *a; const CPUArray *b; const int N = cast(A, B, &a, &b);
	for (int i=0; i<N; ++i) {
		a->data[i] /= b->data[i];
	}
}

void CPUDevice::scale(Array *A, double factor)
{
	CPUArray *a; const int N = cast(A, &a);
	for (int i=0; i<N; ++i) {
		a->data[i] *= factor;
	}
}

void CPUDevice::randomize(Array *A)
{
	CPUArray *a; const int N = cast(A, &a);
	for (int i=0; i<N; ++i) {
		a->data[i] = std::rand() / double(RAND_MAX);
	}
}

double CPUDevice::minimum(const Array *A)
{
	const CPUArray *a; const int N = cast(A, &a);
	double min = +DBL_MAX;
	for (int i=0; i<N; ++i) {
		if (a->data[i] < min) min = a->data[i];
	}
	return min;
}

double CPUDevice::maximum(const Array *A)
{
	const CPUArray *a; const int N = cast(A, &a);
	double max = -DBL_MAX;
	for (int i=0; i<N; ++i) {
		if (a->data[i] > max) max = a->data[i];
	}
	return max;
}

double CPUDevice::average(const Array *A)
{
	return sum(A) / A->getShape().getNumEl();
}

double CPUDevice::sum(const Array *A)
{
	const CPUArray *a; const int N = cast(A, &a);
	double sum = 0.0;
	for (int i=0; i<N; ++i) {
		sum += a->data[i];
	}
	return sum;
}

double CPUDevice::dot(const Array *A, const Array *B)
{
	const CPUArray *a, *b; const int N = cast(A, B, &a, &b);
	double sum = 0.0;
	for (int i=0; i<N; ++i) {
		sum += a->data[i] * b->data[i];
	}
	return sum;
}

void CPUDevice::normalize3(Array *X0, Array *X1, Array *X2, double len)
{
	CPUArray *x0, *x1, *x2;
	const int N0 = cast(X0, &x0);
	const int N1 = cast(X1, &x1);
	const int N2 = cast(X2, &x2);
	assert(N0 == N1 && N1 == N2);

	for (int i=0; i<N0; ++i) {
		const float e0 = x0->data[i];
		const float e1 = x1->data[i];
		const float e2 = x2->data[i];
		
		const double norm = std::sqrt(e0*e0+e1*e1+e2*e2);
		if (norm == 0.0) {
			x0->data[i] = 0.0;
			x1->data[i] = 0.0;
			x2->data[i] = 0.0;
		} else {
			const double scale = len / norm;
			x0->data[i] *= scale;
			x1->data[i] *= scale;
			x2->data[i] *= scale;
		}
	}
}

void CPUDevice::normalize3(Array *X0, Array *X1, Array *X2, const Array *LEN)
{
	CPUArray *x0, *x1, *x2;
	const CPUArray *len;
	const int N0 = cast(X0, &x0);
	const int N1 = cast(X1, &x1);
	const int N2 = cast(X2, &x2);
	const int N3 = cast(LEN, &len);
	assert(N0 == N1 && N1 == N2 && N2 == N3);

	for (int i=0; i<N0; ++i) {
		const float e0 = x0->data[i];
		const float e1 = x1->data[i];
		const float e2 = x2->data[i];
		
		const double norm = std::sqrt(e0*e0+e1*e1+e2*e2);
		if (norm == 0.0) {
			/*x0->data[i] = 0.0;
			x1->data[i] = 0.0;
			x2->data[i] = 0.0;*/
		} else {
			const double scale = len->data[i] / norm;
			x0->data[i] *= scale;
			x1->data[i] *= scale;
			x2->data[i] *= scale;
		}
	}
}

double CPUDevice::absmax3(const Array *X0, const Array *X1, const Array *X2)
{
	const CPUArray *x0, *x1, *x2;
	const int N0 = cast(X0, &x0);
	const int N1 = cast(X1, &x1);
	const int N2 = cast(X2, &x2);
	assert(N0 == N1 && N1 == N2);

	double max_sq = -DBL_MAX;
	for (int i=0; i<N0; ++i) {
		const double x = x0->data[i], y = x1->data[i], z = x2->data[i];
		const double abs_sq = x*x + y*y + z*z;
		if (abs_sq > max_sq) max_sq = abs_sq;
	}
	return std::sqrt(max_sq);
}

double CPUDevice::sumdot3(const Array *X0, const Array *X1, const Array *X2, 
                          const Array *Y0, const Array *Y1, const Array *Y2)
{
	const CPUArray *x0, *x1, *x2;
	const int N0 = cast(X0, &x0);
	const int N1 = cast(X1, &x1);
	const int N2 = cast(X2, &x2);

	const CPUArray *y0, *y1, *y2;
	const int N3 = cast(Y0, &y0);
	const int N4 = cast(Y1, &y1);
	const int N5 = cast(Y2, &y2);

	assert(N0 == N1 && N1 == N2 && N2 == N3 && N3 == N4 && N4 == N5);

	double sum = 0.0;
	for (int i=0; i<N0; ++i) {
		const double a = x0->data[i], c = x1->data[i], e = x2->data[i];
		const double b = y0->data[i], d = y1->data[i], f = y2->data[i];
		const double dot = a*b + c*d + e*f;
		sum += dot;
	}
	return sum;
}

static int cast(Array *arr, CPUArray **cpu_arr)
{
	*cpu_arr = dynamic_cast<CPUArray*>(arr);
	return arr->getShape().getNumEl();
}

static int cast(const Array *arr, const CPUArray **cpu_arr)
{
	*cpu_arr = dynamic_cast<const CPUArray*>(arr);
	return arr->getShape().getNumEl();
}

static int cast(Array *arr1, const Array *arr2, CPUArray **cpu_arr1, const CPUArray **cpu_arr2)
{
	*cpu_arr1 = dynamic_cast<      CPUArray*>(arr1);
	*cpu_arr2 = dynamic_cast<const CPUArray*>(arr2);
	assert(arr1->getShape().getNumEl() == arr2->getShape().getNumEl());
	return arr1->getShape().getNumEl();
}

static int cast(const Array *arr1, const Array *arr2, const CPUArray **cpu_arr1, const CPUArray **cpu_arr2)
{
	*cpu_arr1 = dynamic_cast<const CPUArray*>(arr1);
	*cpu_arr2 = dynamic_cast<const CPUArray*>(arr2);
	assert(arr1->getShape().getNumEl() == arr2->getShape().getNumEl());
	return arr1->getShape().getNumEl();
}

} // ns
