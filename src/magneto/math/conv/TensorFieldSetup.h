#ifndef TENSOR_FIELD_SETUP_H
#define TENSOR_FIELD_SETUP_H

#include "matrix/matty.h"

class TensorFieldSetup
{
public:
	TensorFieldSetup(int num_entries, int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z);
	~TensorFieldSetup();

	ComplexMatrix transformTensorField(const Matrix &tensor);

	void unpackTransformedTensorField_xyz_to_zxy(ComplexMatrix &tensor, Matrix **real_out, Matrix **imag_out);
	void unpackTransformedTensorField_xyz_to_yzx(ComplexMatrix &tensor, Matrix **real_out, Matrix **imag_out);

private:
	const int num_entries; // 6 for symmetric tensors, 3 for antisymmetric tensors
	const int dim_x, dim_y, dim_z;
	const int exp_x, exp_y, exp_z;
};

#endif
