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

#include "TensorFieldSetup.h"

#include "matrix/matty.h"
#include "matrix/matty_ext.h"

TensorFieldSetup::TensorFieldSetup(int num_entries, int dim_x, int dim_y, int dim_z, int exp_x, int exp_y, int exp_z)
	: num_entries(num_entries), dim_x(dim_x), dim_y(dim_y), dim_z(dim_z), exp_x(exp_x), exp_y(exp_y), exp_z(exp_z)
{
}

TensorFieldSetup::~TensorFieldSetup()
{
}

// In and out matrices must have shape (num_entries, exp_x, exp_y, exp_z).
ComplexMatrix TensorFieldSetup::transformTensorField(const Matrix &tensor)
{
	// convert to complex and transform
	Matrix::ro_accessor tensor_acc(tensor);
	ComplexMatrix result(Shape(num_entries, exp_x, exp_y, exp_z));

	// Convert to complex
	{
		ComplexMatrix::accessor result_acc(result);
		for (int n=0; n<result.getShape().getNumEl(); ++n) {
			result_acc.real(n) = tensor_acc.at(n);
			result_acc.imag(n) = 0.0;
		}
	}

	// 3d-transform each component e of (e, exp_x, exp_y, exp_z).
	const int loop_dims[] = {0};
	matty_ext::fftn(result, std::vector<int>(loop_dims, loop_dims+1));
	return result;
}

void TensorFieldSetup::unpackTransformedTensorField_xyz_to_zxy(ComplexMatrix &tensor, Matrix **real_out, Matrix **imag_out)
{
	const double scale = 1.0 / (exp_x * exp_y * exp_z);

	Matrix::wo_accessor *real_acc[20];
	Matrix::wo_accessor *imag_acc[20];

	for (int e=0; e<num_entries; ++e) {
		real_acc[e] = new Matrix::wo_accessor(*(real_out[e]));
		imag_acc[e] = new Matrix::wo_accessor(*(imag_out[e]));
	}

	ComplexMatrix::const_accessor tensor_acc(tensor);
	for (int z=0; z<exp_z; ++z) {
		for (int y=0; y<exp_y; ++y) {
			for (int x=0; x<exp_x/2+1; ++x) {
				for (int e=0; e<num_entries; ++e) {
					real_acc[e]->at(z,x,y) = scale * tensor_acc.real(e,x,y,z);
					imag_acc[e]->at(z,x,y) = scale * tensor_acc.imag(e,x,y,z);
				}
			}
		}
	}

	for (int e=0; e<num_entries; ++e) {
		delete real_acc[e];
		delete imag_acc[e];
	}
}

void TensorFieldSetup::unpackTransformedTensorField_xyz_to_yzx(ComplexMatrix &tensor, Matrix **real_out, Matrix **imag_out)
{
	const double scale = 1.0 / (exp_x * exp_y * exp_z);

	Matrix::wo_accessor *real_acc[20];
	Matrix::wo_accessor *imag_acc[20];

	for (int e=0; e<num_entries; ++e) {
		real_acc[e] = new Matrix::wo_accessor(*(real_out[e]));
		imag_acc[e] = new Matrix::wo_accessor(*(imag_out[e]));
	}

	ComplexMatrix::const_accessor tensor_acc(tensor);
	for (int z=0; z<exp_z; ++z) {
		for (int y=0; y<exp_y; ++y) {
			for (int x=0; x<exp_x/2+1; ++x) {
				for (int e=0; e<num_entries; ++e) {
					real_acc[e]->at(y,z,x) = scale * tensor_acc.real(e,x,y,z);
					imag_acc[e]->at(y,z,x) = scale * tensor_acc.imag(e,x,y,z);
				}
			}
		}
	}

	for (int e=0; e<num_entries; ++e) {
		delete real_acc[e];
		delete imag_acc[e];
	}
}
