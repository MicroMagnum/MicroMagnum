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

#ifndef MATRIX_H
#define MATRIX_H

#include "config.h"

#include "matrix/AbstractMatrix.h"

#include "Matrix_accessor.h"
#ifdef HAVE_CUDA
#include "Matrix_cuda_accessor.h"
#endif

namespace matty {

class Matrix : public AbstractMatrix
{
public:
	typedef MatrixAccessor_read_only  ro_accessor;
	typedef MatrixAccessor_write_only wo_accessor;
	typedef MatrixAccessor_read_write rw_accessor;

#ifdef HAVE_CUDA
	typedef      MatrixCU32Accessor cuda_accessor;
	typedef ConstMatrixCU32Accessor const_cuda_accessor;
	typedef      MatrixCU32Accessor cu32_accessor;
	typedef ConstMatrixCU32Accessor const_cu32_accessor;
#ifdef HAVE_CUDA_64
	typedef      MatrixCU64Accessor cu64_accessor;
	typedef ConstMatrixCU64Accessor const_cu64_accessor;
#endif
#endif

	Matrix();
	Matrix(const Shape &shape);
	Matrix(const Matrix &other);
	Matrix &operator=(Matrix other);
	virtual ~Matrix();

	void clear();
	void fill(double value);
	void assign(const Matrix &other);
	void scale(double factor);
	void add(const Matrix &op, double scale = 1.0);
	void multiply(const Matrix &rhs);
	void divide(const Matrix &rhs);
	void randomize();

	double maximum() const;
	double minimum() const;
	double average() const;
	double sum() const;

	double absMax() const;

	double getUniformValue() const;

	Array *getArray(int dev) const;
};

inline Matrix zeros(const Shape &shape)
{
	Matrix a(shape); a.fill(0.0); return a;
}

inline Matrix ones(const Shape &shape)
{
	Matrix a(shape); a.fill(1.0); return a;
}

std::ostream &operator<<(std::ostream &out, const Matrix &mat);

} // ns

#endif
