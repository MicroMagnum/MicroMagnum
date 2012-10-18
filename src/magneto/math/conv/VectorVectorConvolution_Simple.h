#ifndef VECTOR_VECTOR_CONVOLUTION_SIMPLE_H
#define VECTOR_VECTOR_CONVOLUTION_SIMPLE_H

#include "VectorVectorConvolution.h"

#include "matrix/matty.h"

class VectorVectorConvolution_Simple
{
public:
	VectorVectorConvolution_Simple(const VectorMatrix &lhs, int dim_x, int dim_y, int dim_z);
	~VectorVectorConvolution_Simple();

	virtual void execute(const VectorMatrix &rhs, Matrix &res);

private:
	VectorMatrix lhs;
	int dim_x, dim_y, dim_z;
	int exp_x, exp_y, exp_z;
};

#endif
