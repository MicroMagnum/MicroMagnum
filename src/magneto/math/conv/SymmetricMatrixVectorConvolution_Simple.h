#ifndef SYMMETRIC_MATRIX_VECTOR_CONVOLUTION_SIMPLE_H
#define SYMMETRIC_MATRIX_VECTOR_CONVOLUTION_SIMPLE_H

#include "matrix/matty.h"

class SymmetricMatrixVectorConvolution_Simple
{
public:
	SymmetricMatrixVectorConvolution_Simple(const Matrix &lhs, int dim_x, int dim_y, int dim_z);
	~SymmetricMatrixVectorConvolution_Simple();

	virtual void execute(const VectorMatrix &rhs, VectorMatrix &res);

private:
	Matrix lhs;
	int dim_x, dim_y, dim_z;
	int exp_x, exp_y, exp_z;
};

#endif
