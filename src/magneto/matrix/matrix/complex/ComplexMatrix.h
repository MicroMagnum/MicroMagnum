/*
 * Copyright 2012 by the Micromagnum authors.
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

#ifndef COMPLEX_MATRIX_H
#define COMPLEX_MATRIX_H

#include "config.h"
#include "matrix/AbstractMatrix.h"
#include "matrix/scalar/Matrix.h"

#include <complex>

namespace matty {

class      ComplexMatrixAccessor;
class ConstComplexMatrixAccessor;

class Matrix;

class ComplexMatrix : public AbstractMatrix
{
public:
	typedef      ComplexMatrixAccessor accessor;
	typedef ConstComplexMatrixAccessor const_accessor;

	ComplexMatrix(const Shape &shape);
	ComplexMatrix(const ComplexMatrix &other);
	ComplexMatrix &operator=(ComplexMatrix other);
	virtual ~ComplexMatrix();

	void clear();
	void fill(double real, double imag);
	void fill(std::complex<double> value);
	//void assign(const Matrix &real, const Matrix &imag);
	void assign(const ComplexMatrix &other);
	void randomize();

	std::complex<double> getUniformValue() const;

	Array *getArray(int dev, int comp) const;
};

std::ostream &operator<<(std::ostream &out, const ComplexMatrix &mat);

} // ns

#include "ComplexMatrix_accessor.h"

#endif
