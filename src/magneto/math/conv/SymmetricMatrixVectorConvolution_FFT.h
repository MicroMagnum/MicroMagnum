#ifndef SYMMETRIC_MATRIX_VECTOR_CONVOLUTION_FFT_H
#define SYMMETRIC_MATRIX_VECTOR_CONVOLUTION_FFT_H

#include "MatrixVectorConvolution_FFT.h"

class SymmetricMatrixVectorConvolution_FFT : public MatrixVectorConvolution_FFT
{
public:
	SymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~SymmetricMatrixVectorConvolution_FFT();

private:
	virtual void calculate_multiplication(double *inout_x, double *inout_y, double *inout_z);
#ifdef HAVE_CUDA
	virtual void calculate_multiplication_cuda(float *inout_x, float *inout_y, float *inout_z);
#endif

	// Buffers
	struct tensor_buf {
		Matrix re[6], im[6];
	} N;
};

#endif
