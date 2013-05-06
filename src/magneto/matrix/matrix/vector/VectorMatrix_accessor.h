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

#ifndef VECTOR_MATRIX_ACCESSOR_H
#define VECTOR_MATRIX_ACCESSOR_H

#include "config.h"
#include "restrict.h"
#include "Vector3d.h"

namespace matty {

class VectorMatrix;

class VectorMatrixAccessor
{
public:
	VectorMatrixAccessor(VectorMatrix &mat);
	~VectorMatrixAccessor();

	void set(int x0,                         Vector3d value) { data_x[x0] = value.x; data_y[x0] = value.y; data_z[x0] = value.z; }
	void set(int x0, int x1,                 Vector3d value) { set(x0 + x1*strides[1], value); }
	void set(int x0, int x1, int x2,         Vector3d value) { set(x0 + x1*strides[1] + x2*strides[2], value); }
	void set(int x0, int x1, int x2, int x3, Vector3d value) { set(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3], value); }

	Vector3d get(int x0                        ) const { return Vector3d(data_x[x0], data_y[x0], data_z[x0]); }
	Vector3d get(int x0, int x1                ) const { return get(x0 + x1*strides[1]); }
	Vector3d get(int x0, int x1, int x2        ) const { return get(x0 + x1*strides[1] + x2*strides[2]); }
	Vector3d get(int x0, int x1, int x2, int x3) const { return get(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3]); }

	double *ptr_x() const { return data_x; }
	double *ptr_y() const { return data_y; }
	double *ptr_z() const { return data_z; }

private:
	VectorMatrix &mat;
	double * RESTRICT data_x, * RESTRICT data_y, * RESTRICT data_z;
	int strides[4]; // strides of the first 4 dimensions
};

class ConstVectorMatrixAccessor
{
public:
	ConstVectorMatrixAccessor(const VectorMatrix &mat);
	~ConstVectorMatrixAccessor();

	Vector3d get(int x0                        ) const { return Vector3d(data_x[x0], data_y[x0], data_z[x0]); }
	Vector3d get(int x0, int x1                ) const { return get(x0 + x1*strides[1]); }
	Vector3d get(int x0, int x1, int x2        ) const { return get(x0 + x1*strides[1] + x2*strides[2]); }
	Vector3d get(int x0, int x1, int x2, int x3) const { return get(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3]); }

	const double *ptr_x() const { return data_x; }
	const double *ptr_y() const { return data_y; }
	const double *ptr_z() const { return data_z; }

private:
	const VectorMatrix &mat;
	const double * RESTRICT data_x, * RESTRICT data_y, * RESTRICT data_z;
	int strides[4]; // strides of the first 4 dimensions
};
	
} // ns

#endif
