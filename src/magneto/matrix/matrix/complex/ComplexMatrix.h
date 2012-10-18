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
