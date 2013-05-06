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

#ifndef MAGNETO_CUBLAS_WRAP_H
#define MAGNETO_CUBLAS_WRAP_H

#include "config.h"
#include <cublas.h>

// Overloaded function wrappers for 32-float and 64-double data (if enabled).
// Only the wrappers that are actually used are provided.
namespace cublas_wrap
{
	inline void copy(int n, const float *x, int incx, float *y, int incy) { cublasScopy(n, x, incx, y, incy); }
	inline float dot(int n, const float *x, int incx, const float *y, int incy) { return cublasSdot(n, x, incx, y, incy); }
	inline void axpy(int n, float alpha, const float *x, int incx, float *y, int incy) { cublasSaxpy(n, alpha, x, incx, y, incy); }
	inline void scal(int n, float alpha, float *x, int incx) { cublasSscal(n, alpha, x, incx); }

#ifdef HAVE_CUDA_64
	inline void copy(int n, const double *x, int incx, double *y, int incy) { cublasDcopy(n, x, incx, y, incy); }
	inline double dot(int n, const double *x, int incx, const double *y, int incy) { return cublasDdot(n, x, incx, y, incy); }
	inline void axpy(int n, double alpha, const double *x, int incx, double *y, int incy) { cublasDaxpy(n, alpha, x, incx, y, incy); }
	inline void scal(int n, double alpha, double *x, int incx) { cublasDscal(n, alpha, x, incx); }
#endif
}

#endif
