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

#include "cublas_wrap.h"
#include "kernels_simple.h"
#include "kernels_reduce.h"

#include <stdexcept>
#include <cassert>

namespace matty {

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::clear(Array *A) // A[*] <- 0
	{
		CUDAArrayT *a; const int N = cast(A, &a);
		cuda_fill(a->ptr(), 0, N);
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::fill(Array *A, double value) // A[*] <- value
	{
		CUDAArrayT *a; const int N = cast(A, &a);
		cuda_fill(a->ptr(), value, N);
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::assign(Array *A, const Array *B) // A[n] <- B[n]
	{
		CUDAArrayT *a; const CUDAArrayT *b; 
		const int N = cast(A, B, &a, &b);
		cublas_wrap::copy(N, b->ptr(), 1, a->ptr(), 1);
		if (cublasGetError() != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("CUDADeviceMixin::assign: cublas copy returned error code");
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::add(Array *A, const Array *B, double scale) // A[n] <- B[n] * scale
	{
		CUDAArrayT *a; const CUDAArrayT *b; 
		const int N = cast(A, B, &a, &b);
		cublas_wrap::axpy(N, scale, b->ptr(), 1, a->ptr(), 1);
		if (cublasGetError() != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("CUDADeviceMixin::add: cublas saxpy returned error code");
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::multiply(Array *A, const Array *B)
	{
		CUDAArrayT *a; const CUDAArrayT *b; 
		const int N = cast(A, B, &a, &b);
		cuda_mul(a->ptr(), b->ptr(), N);
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::divide(Array *A, const Array *B)
	{
		CUDAArrayT *a; const CUDAArrayT *b; 
		const int N = cast(A, B, &a, &b);
		cuda_div(a->ptr(), b->ptr(), N);
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::scale(Array *A, double factor)
	{
		CUDAArrayT *a; 
		const int N = cast(A, &a);
		cublas_wrap::scal(N, factor, a->ptr(), 1);
		if (cublasGetError() != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("CUDADeviceMixin::scale: cublas saxpy returned error code");
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::randomize(Array *A)
	{
		//CUDAArrayT *a; const int N = cast(A, &a);
		assert("Not implemented: CUDADeviceMixin::randomize" && 0);
	}

	template <class CUDAArrayT>
	double CUDADeviceMixin<CUDAArrayT>::minimum(const Array *A)
	{
		const CUDAArrayT *a; 
		const int N = cast(A, &a);
		return cuda_min(a->ptr(), N);
	}

	template <class CUDAArrayT>
	double CUDADeviceMixin<CUDAArrayT>::maximum(const Array *A)
	{
		const CUDAArrayT *a; 
		const int N = cast(A, &a);
		return cuda_max(a->ptr(), N);
	}

	template <class CUDAArrayT>
	double CUDADeviceMixin<CUDAArrayT>::sum(const Array *A)
	{
		const CUDAArrayT *a; 
		const int N = cast(A, &a);
		return cuda_sum(a->ptr(), N);
	}

	template <class CUDAArrayT>
	double CUDADeviceMixin<CUDAArrayT>::average(const Array *A)
	{
		return sum(A) / A->getShape().getNumEl();
	}

	template <class CUDAArrayT>
	double CUDADeviceMixin<CUDAArrayT>::dot(const Array *A, const Array *B)
	{
		const CUDAArrayT *a, *b; 
		const int N = cast(A, B, &a, &b);
		const double dot = cublas_wrap::dot(N, b->ptr(), 1, a->ptr(), 1);
		if (cublasGetError() != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("CUDADeviceMixin::dot: cublas dot returned error code");
		return dot;
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::normalize3(Array *X0, Array *X1, Array *X2, double len)
	{
		CUDAArrayT *x0, *x1, *x2;
		const int N0 = cast(X0, &x0);
		const int N1 = cast(X1, &x1);
		const int N2 = cast(X2, &x2);
		assert(N0 == N1 && N1 == N2);
		cuda_normalize3(x0->ptr(), x1->ptr(), x2->ptr(), len, N0);
	}

	template <class CUDAArrayT>
	void CUDADeviceMixin<CUDAArrayT>::normalize3(Array *X0, Array *X1, Array *X2, const Array *LEN)
	{
		CUDAArrayT *x0, *x1, *x2;
		const CUDAArrayT *len;
		const int N0 = cast(X0, &x0);
		const int N1 = cast(X1, &x1);
		const int N2 = cast(X2, &x2);
		const int N3 = cast(LEN, &len);
		assert(N0 == N1 && N1 == N2 && N2 == N3);
		cuda_normalize3(x0->ptr(), x1->ptr(), x2->ptr(), len->ptr(), N0);
	}

	template <class CUDAArrayT>
	double CUDADeviceMixin<CUDAArrayT>::absmax3(const Array *X0, const Array *X1, const Array *X2)
	{
		const CUDAArrayT *x0, *x1, *x2;
		const int N0 = cast(X0, &x0);
		const int N1 = cast(X1, &x1);
		const int N2 = cast(X2, &x2);
		assert(N0 == N1 && N1 == N2);
		return cuda_absmax3(x0->ptr(), x1->ptr(), x2->ptr(), N0);
	}

	template <class CUDAArrayT>
	double CUDADeviceMixin<CUDAArrayT>::sumdot3(const Array *X0, const Array *X1, const Array *X2, 
				   const Array *Y0, const Array *Y1, const Array *Y2)
	{
		const CUDAArrayT *x0, *x1, *x2;
		const int N0 = cast(X0, &x0);
		const int N1 = cast(X1, &x1);
		const int N2 = cast(X2, &x2);

		const CUDAArrayT *y0, *y1, *y2;
		const int N3 = cast(Y0, &y0);
		const int N4 = cast(Y1, &y1);
		const int N5 = cast(Y2, &y2);

		assert(N0 == N1 && N1 == N2 && N2 == N3 && N4 == N5);
		return cuda_sumdot3(x0->ptr(), x1->ptr(), x2->ptr(), y0->ptr(), y1->ptr(), y2->ptr(), N0);
	}

	template <class CUDAArrayT>
	int CUDADeviceMixin<CUDAArrayT>::cast(Array *arr, CUDAArrayT **cuda_arr)
	{
		*cuda_arr = dynamic_cast<CUDAArrayT*>(arr);
		return arr->getShape().getNumEl();
	}

	template <class CUDAArrayT>
	int CUDADeviceMixin<CUDAArrayT>::cast(const Array *arr, const CUDAArrayT **cuda_arr)
	{
		*cuda_arr = dynamic_cast<const CUDAArrayT*>(arr);
		return arr->getShape().getNumEl();
	}

	template <class CUDAArrayT>
	int CUDADeviceMixin<CUDAArrayT>::cast(Array *arr1, const Array *arr2, CUDAArrayT **cuda_arr1, const CUDAArrayT **cuda_arr2)
	{
		*cuda_arr1 = dynamic_cast<      CUDAArrayT*>(arr1);
		*cuda_arr2 = dynamic_cast<const CUDAArrayT*>(arr2);
		assert(arr1->getShape().getNumEl() == arr2->getShape().getNumEl());
		return arr1->getShape().getNumEl();
	}

	template <class CUDAArrayT>
	int CUDADeviceMixin<CUDAArrayT>::cast(const Array *arr1, const Array *arr2, const CUDAArrayT **cuda_arr1, const CUDAArrayT **cuda_arr2)
	{
		*cuda_arr1 = dynamic_cast<const CUDAArrayT*>(arr1);
		*cuda_arr2 = dynamic_cast<const CUDAArrayT*>(arr2);
		assert(arr1->getShape().getNumEl() == arr2->getShape().getNumEl());
		return arr1->getShape().getNumEl();
	}

}
