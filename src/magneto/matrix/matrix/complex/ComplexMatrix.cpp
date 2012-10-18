#include "config.h"
#include "ComplexMatrix.h"
#include "matty.h"

#include <stdexcept>
#include <cassert>
#include <cstddef>

#include <iostream>
using namespace std;

namespace matty {

ComplexMatrix::ComplexMatrix(const Shape &shape)
	: AbstractMatrix(shape, 2)
{
}

ComplexMatrix::ComplexMatrix(const ComplexMatrix &other)
	: AbstractMatrix(other)
{
}

ComplexMatrix &ComplexMatrix::operator=(ComplexMatrix other)
{
	swap(other);
	return *this;
}

ComplexMatrix::~ComplexMatrix()
{
}

std::complex<double> ComplexMatrix::getUniformValue() const
{
	if (!isUniform()) throw std::runtime_error("Cant get uniform value of matrix because matrix is not uniform");
	return std::complex<double>(uval[0], uval[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void ComplexMatrix::clear()
{
	fill(0.0, 0.0);
}

void ComplexMatrix::fill(double real, double imag)
{
	if (isLocked()) throw std::runtime_error("Can't fill/clear matrix because matrix is already locked!");

	// We need to invalidate arrays if they do not uniformly contain 'value'
	const bool need_to_invalidate = (state != UNIFORM || uval[0] != real || uval[1] != imag);
	if (need_to_invalidate) {
		for (size_t i=0; i<info.size(); ++i)
			info[i].valid = false;
	}

	uval[0] = real;
	uval[1] = imag;
	state = UNIFORM;
}

void ComplexMatrix::fill(std::complex<double> value)
{
	fill(value.real(), value.imag());
}

void ComplexMatrix::assign(const ComplexMatrix &op)
{
	if (this == &op) {
		return;
	} else if (op.isUniform()) {
		fill(op.uval[0], op.uval[1]);
	} else {
		const int dev = computeStrategy2(op);
		writeLock(dev); op.readLock(dev);
		matty::getDevice(dev)->assign(getArray(dev, 0), op.getArray(dev, 0));
		matty::getDevice(dev)->assign(getArray(dev, 1), op.getArray(dev, 1));
		writeUnlock(dev); op.readUnlock(dev);
	}
}

void ComplexMatrix::randomize()
{
	const int dev = computeStrategy1();
	writeLock(dev);
	matty::getDevice(dev)->randomize(getArray(dev, 0));
	matty::getDevice(dev)->randomize(getArray(dev, 1));
	writeUnlock(dev);
}

Array *ComplexMatrix::getArray(int dev, int comp) const
{
	return info[dev].arrays[comp];
}

std::ostream &operator<<(std::ostream &out, const ComplexMatrix &mat)
{
	out << "ComplexMatrix@" << (void*)&mat << endl;
	return out;
}

} // ns
