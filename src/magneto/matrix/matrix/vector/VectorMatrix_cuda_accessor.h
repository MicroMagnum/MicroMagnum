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

#ifndef VECTOR_MATRIX_CUDA_ACCESSOR_H
#define VECTOR_MATRIX_CUDA_ACCESSOR_H

#include "config.h"

namespace matty {

class VectorMatrix;

class VectorMatrixCU32Accessor
{
public:
	VectorMatrixCU32Accessor(VectorMatrix &mat);
	~VectorMatrixCU32Accessor();

	float *ptr_x() const { return data_x; }
	float *ptr_y() const { return data_y; }
	float *ptr_z() const { return data_z; }

private:
	VectorMatrix &mat;
	float *data_x, *data_y, *data_z; // cuda device pointers..
};

class ConstVectorMatrixCU32Accessor
{
public:
	ConstVectorMatrixCU32Accessor(const VectorMatrix &mat);
	~ConstVectorMatrixCU32Accessor();

	const float *ptr_x() const { return data_x; }
	const float *ptr_y() const { return data_y; }
	const float *ptr_z() const { return data_z; }

private:
	const VectorMatrix &mat;
	const float *data_x, *data_y, *data_z; // cuda device pointers..
};

#ifdef HAVE_CUDA_64
class VectorMatrixCU64Accessor
{
public:
	VectorMatrixCU64Accessor(VectorMatrix &mat);
	~VectorMatrixCU64Accessor();

	double *ptr_x() const { return data_x; }
	double *ptr_y() const { return data_y; }
	double *ptr_z() const { return data_z; }

private:
	VectorMatrix &mat;
	double *data_x, *data_y, *data_z; // cuda device pointers..
};

class ConstVectorMatrixCU64Accessor
{
public:
	ConstVectorMatrixCU64Accessor(const VectorMatrix &mat);
	~ConstVectorMatrixCU64Accessor();

	const double *ptr_x() const { return data_x; }
	const double *ptr_y() const { return data_y; }
	const double *ptr_z() const { return data_z; }

private:
	const VectorMatrix &mat;
	const double *data_x, *data_y, *data_z; // cuda device pointers..
};
#endif

// Structs to help with writing generic code.
template <typename real>
struct VectorMatrix_cuda_accessor {};

template <typename real>
struct VectorMatrix_const_cuda_accessor {};

template <>
struct VectorMatrix_cuda_accessor<float> {
	typedef VectorMatrixCU32Accessor t;
};

template <>
struct VectorMatrix_const_cuda_accessor<float> {
	typedef ConstVectorMatrixCU32Accessor t;
};
#ifdef HAVE_CUDA_64
template <>
struct VectorMatrix_cuda_accessor<double> {
	typedef VectorMatrixCU64Accessor t;
};

template <>
struct VectorMatrix_const_cuda_accessor<double> {
	typedef ConstVectorMatrixCU64Accessor t;
};
#endif

} // ns

#endif
