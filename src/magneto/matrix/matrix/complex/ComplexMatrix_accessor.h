#ifndef COMPLEX_MATRIX_ACCESSOR_H
#define COMPLEX_MATRIX_ACCESSOR_H

#include "config.h"
#include "restrict.h"

#include "ComplexMatrix.h"
#include "device/cpu/CPUArray.h"

namespace matty {

class ComplexMatrixAccessor
{
public:
	ComplexMatrixAccessor(ComplexMatrix &mat) : mat(mat) 
	{
		mat.writeLock(0); // 0 = CPUDevice!
		data_re = static_cast<CPUArray*>(mat.getArray(0, 0))->ptr();
		data_im = static_cast<CPUArray*>(mat.getArray(0, 1))->ptr();
	
		// Precalculate strides
		const int rank = mat.getShape().getRank();
		strides[0] = 1;
		strides[1] = strides[0] * (rank > 0 ? mat.getShape().getDim(0) : 1);
		strides[2] = strides[1] * (rank > 1 ? mat.getShape().getDim(1) : 1);
		strides[3] = strides[2] * (rank > 2 ? mat.getShape().getDim(2) : 1);
	}

	~ComplexMatrixAccessor()
	{
		mat.writeUnlock(0);
	}

	double &real(int x0) { return data_re[x0]; }
	double &real(int x0, int x1) { return real(x0 + x1*strides[1]); }
	double &real(int x0, int x1, int x2) { return real(x0 + x1*strides[1] + x2*strides[2]); }
	double &real(int x0, int x1, int x2, int x3) { return real(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3]); }

	double &imag(int x0) { return data_im[x0]; }
	double &imag(int x0, int x1) { return imag(x0 + x1*strides[1]); }
	double &imag(int x0, int x1, int x2) { return imag(x0 + x1*strides[1] + x2*strides[2]); }
	double &imag(int x0, int x1, int x2, int x3) { return imag(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3]); }

	double *ptr_real() const { return data_re; }
	double *ptr_imag() const { return data_im; }

private:
	ComplexMatrix &mat;
	double * RESTRICT data_re, * RESTRICT data_im;
	int strides[4]; // strides of the first 4 dimensions
};

class ConstComplexMatrixAccessor
{
public:
	ConstComplexMatrixAccessor(const ComplexMatrix &mat) : mat(mat) 
	{
		mat.readLock(0);
		data_re = static_cast<CPUArray*>(mat.getArray(0, 0))->ptr();
		data_im = static_cast<CPUArray*>(mat.getArray(0, 1))->ptr();
	
		// Precalculate strides
		const int rank = mat.getShape().getRank();
		strides[0] = 1;
		strides[1] = strides[0] * (rank > 0 ? mat.getShape().getDim(0) : 1);
		strides[2] = strides[1] * (rank > 1 ? mat.getShape().getDim(1) : 1);
		strides[3] = strides[2] * (rank > 2 ? mat.getShape().getDim(2) : 1);
	}

	~ConstComplexMatrixAccessor()
	{
		mat.readUnlock(0);
	}

	const double &real(int x0) { return data_re[x0]; }
	const double &real(int x0, int x1) { return real(x0 + x1*strides[1]); }
	const double &real(int x0, int x1, int x2) { return real(x0 + x1*strides[1] + x2*strides[2]); }
	const double &real(int x0, int x1, int x2, int x3) { return real(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3]); }

	const double &imag(int x0) { return data_im[x0]; }
	const double &imag(int x0, int x1) { return imag(x0 + x1*strides[1]); }
	const double &imag(int x0, int x1, int x2) { return imag(x0 + x1*strides[1] + x2*strides[2]); }
	const double &imag(int x0, int x1, int x2, int x3) { return imag(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3]); }

	const double *ptr_real() const { return data_re; }
	const double *ptr_imag() const { return data_im; }

private:
	const ComplexMatrix &mat;
	const double * RESTRICT data_re, * RESTRICT data_im;
	int strides[4]; // strides of the first 4 dimensions
};
	
} // ns

#endif
