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
