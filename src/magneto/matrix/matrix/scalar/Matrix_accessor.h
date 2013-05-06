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

#ifndef MATRIX_ACCESSOR_H
#define MATRIX_ACCESSOR_H

#include "config.h"
#include "restrict.h"
#include "matrix/Shape.h"

namespace matty {

class Matrix;

#define MATRIX_ACCESSOR_DEFS(qualifier)                                                                                          \
public:                                                                                                                         \
	qualifier double &at(int x0) { return data[x0]; }                                                                       \
	qualifier double &at(int x0, int x1) { return at(x0 + x1*strides[1]); }                                                 \
	qualifier double &at(int x0, int x1, int x2) { return at(x0 + x1*strides[1] + x2*strides[2]); }                         \
	qualifier double &at(int x0, int x1, int x2, int x3) { return at(x0 + x1*strides[1] + x2*strides[2] + x3*strides[3]); } \
	qualifier double *ptr() { return data; }                                                                                \
private:                                                                                                                        \
	qualifier Matrix &mat;                                                                                                  \
	qualifier double * RESTRICT data;                                                                                        \
	int strides[4];

class MatrixAccessor_read_only // read-only
{
public:
	MatrixAccessor_read_only(const Matrix &mat);
	~MatrixAccessor_read_only();

	MATRIX_ACCESSOR_DEFS(const)
};

class MatrixAccessor_write_only // write-only
{
public:
	MatrixAccessor_write_only(Matrix &mat);
	~MatrixAccessor_write_only();

	MATRIX_ACCESSOR_DEFS()
};

class MatrixAccessor_read_write // read-write
{
public:
	MatrixAccessor_read_write(Matrix &mat);
	~MatrixAccessor_read_write();

	MATRIX_ACCESSOR_DEFS()
};

#undef MATRIX_ACCESSOR_DEFS

} // ns

#endif
