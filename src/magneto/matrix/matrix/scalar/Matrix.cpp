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

#include "config.h"
#include "Matrix.h"
#include "matty.h"

#include <stdexcept>
#include <cassert>
#include <cstddef>
#include <iostream>

namespace matty {

Matrix::Matrix() : AbstractMatrix(Shape(), 1)
{
}

Matrix::Matrix(const Shape &shape)
	: AbstractMatrix(shape, 1)
{
}

Matrix::Matrix(const Matrix &other)
	: AbstractMatrix(other)
{
}

Matrix &Matrix::operator=(Matrix other)
{
	swap(other);
	return *this;
}

Matrix::~Matrix()
{
}

double Matrix::getUniformValue() const
{
	if (!isUniform()) throw std::runtime_error("Cant get uniform value because matrix is not uniform");
	return uval[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Matrix::clear()
{
	fill(0.0);
}

void Matrix::fill(double value)
{
	if (isLocked()) throw std::runtime_error("Can't fill/clear matrix because matrix is already locked!");

	// We need to invalidate arrays if they do not uniformly contain 'value'
	const bool need_to_invalidate = (state != UNIFORM || value != uval[0]);
	if (need_to_invalidate) {
		for (size_t i=0; i<info.size(); ++i)
			info[i].valid = false;
	}

	uval[0] = value;
	state = UNIFORM;
}

void Matrix::assign(const Matrix &op)
{
	if (this == &op) {
		return;
	} else if (op.isUniform()) {
		fill(op.getUniformValue());
	} else {
		const int dev = computeStrategy2(op);
		writeLock(dev); op.readLock(dev);
		matty::getDevice(dev)->assign(getArray(dev), op.getArray(dev));
		writeUnlock(dev); op.readUnlock(dev);
	}
}

void Matrix::scale(double factor)
{
	if (factor == 0.0) {
		fill(0.0);
	} else if (isUniform()) {
		fill(getUniformValue() * factor);
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		matty::getDevice(dev)->scale(getArray(dev), factor);
		readUnlock(dev);
	}
}

void Matrix::add(const Matrix &op, double factor)
{
	if (this == &op) {
		scale(1.0 + factor);
	} else if (isUniform() && op.isUniform()) {
		fill(getUniformValue() + op.getUniformValue() * factor);
	} else {
		const int dev = computeStrategy2(op);
		writeLock(dev); op.readLock(dev);
		matty::getDevice(dev)->add(getArray(dev), op.getArray(dev), factor);
		writeUnlock(dev); op.readUnlock(dev);
	}
}

void Matrix::multiply(const Matrix &op)
{
	if (this == &op) {
		assert(0);
	} else if (op.isUniform()) {
		scale(op.getUniformValue());
	} else {
		const int dev = computeStrategy2(op);
		writeLock(dev); op.readLock(dev);
		matty::getDevice(dev)->multiply(getArray(dev), op.getArray(dev));
		writeUnlock(dev); op.readUnlock(dev);
	}
}

void Matrix::divide(const Matrix &op)
{
	if (this == &op) {
		assert(0);
	} else if (op.isUniform()) {
		scale(1.0 / op.getUniformValue());
	} else {
		const int dev = computeStrategy2(op);
		writeLock(dev); op.readLock(dev);
		matty::getDevice(dev)->divide(getArray(dev), op.getArray(dev));
		writeUnlock(dev); op.readUnlock(dev);
	}
}

double Matrix::maximum() const
{
	if (isUniform()) {
		return getUniformValue();
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		const double max = matty::getDevice(dev)->maximum(getArray(dev));
		readUnlock(dev);
		return max;
	}
}

double Matrix::minimum() const
{
	if (isUniform()) {
		return getUniformValue();
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		const double min = matty::getDevice(dev)->minimum(getArray(dev));
		readUnlock(dev);
		return min;
	}
}

double Matrix::absMax() const
{
	return std::max(std::abs(maximum()), std::abs(minimum()));
}

double Matrix::sum() const
{
	if (isUniform()) {
		return getUniformValue() * getShape().getNumEl();
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		const double sum = matty::getDevice(dev)->sum(getArray(dev));
		readUnlock(dev);
		return sum;
	}
}

Array *Matrix::getArray(int dev) const
{
	return info[dev].arrays[0];
}

void Matrix::randomize()
{
	const int dev = computeStrategy1();
	writeLock(dev);
	matty::getDevice(dev)->randomize(getArray(dev));
	writeUnlock(dev);
}

double Matrix::average() const
{
	if (isUniform()) {
		return getUniformValue();
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		const double avg = matty::getDevice(dev)->average(getArray(dev));
		readUnlock(dev);
		return avg;
	}
}

std::ostream &operator<<(std::ostream &out, const Matrix &mat)
{
	out << "Matrix@" << (void*)&mat << std::endl;
	return out;
}

} // ns
