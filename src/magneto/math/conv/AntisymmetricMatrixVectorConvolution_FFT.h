#ifndef ASYMMETRIC_MATRIX_VECTOR_CONVOLUTION_FFT_H
#define ASYMMETRIC_MATRIX_VECTOR_CONVOLUTION_FFT_H

#include "MatrixVectorConvolution_FFT.h"

class AntisymmetricMatrixVectorConvolution_FFT : public MatrixVectorConvolution_FFT
{
public:
	AntisymmetricMatrixVectorConvolution_FFT(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	virtual ~AntisymmetricMatrixVectorConvolution_FFT();

protected:
	virtual void calculate_multiplication(double *inout_x, double *inout_y, double *inout_z);
#ifdef HAVE_CUDA
	virtual void calculate_multiplication_cuda(float *inout_x, float *inout_y, float *inout_z);
#endif

private:
	// Buffers
	struct tensor_buf {
		Matrix re[3], im[3];
	} N;
};

#endif
