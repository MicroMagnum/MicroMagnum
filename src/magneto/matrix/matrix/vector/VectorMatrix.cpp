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

#include "VectorMatrix.h"
#include "matty.h"

#include <stdexcept>
#include <cstddef>

namespace matty {

VectorMatrix::VectorMatrix() : AbstractMatrix(Shape(), 3)
{
}

VectorMatrix::VectorMatrix(const Shape &shape)
	: AbstractMatrix(shape, 3)
{
}

VectorMatrix::VectorMatrix(const VectorMatrix &other)
	: AbstractMatrix(other)
{
}

VectorMatrix &VectorMatrix::operator=(VectorMatrix other)
{
	swap(other);
	return *this;
}

VectorMatrix::~VectorMatrix()
{
}

Vector3d VectorMatrix::getUniformValue() const
{
	if (!isUniform()) throw std::runtime_error("Can't get uniform value because matrix is not uniform");
	return Vector3d(uval[0], uval[1], uval[2]);
}

Array *VectorMatrix::getArray(int dev, int comp) const
{
	return info[dev].arrays[comp];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void VectorMatrix::clear()
{
	fill(Vector3d(0.0, 0.0, 0.0));
}

void VectorMatrix::fill(Vector3d value)
{
	if (isLocked()) throw std::runtime_error("Can't fill/clear vector matrix because matrix is already locked!");

	// We need to invalidate arrays if they do not uniformly contain 'value'
	const bool need_to_invalidate = (state != UNIFORM || uval[0] != value.x || uval[1] != value.y || uval[2] != value.z);
	if (need_to_invalidate) {
		for (size_t i=0; i<info.size(); ++i)
			info[i].valid = false;
	}

	uval[0] = value.x; uval[1] = value.y; uval[2] = value.z;
	state = UNIFORM;
}

void VectorMatrix::assign(const VectorMatrix &op)
{
	if (this == &op) {
		return;
	} else if (op.isUniform()) {
		fill(op.getUniformValue());
	} else {
		const int dev = computeStrategy2(op);
		writeLock(dev); op.readLock(dev);
		for (int c=0; c<num_arrays; ++c) {
			matty::getDevice(dev)->assign(getArray(dev, c), op.getArray(dev, c));
		}
		writeUnlock(dev); op.readUnlock(dev);
	}
}

void VectorMatrix::scale(double factor)
{
	if (factor == 0.0) {
		fill(Vector3d(0.0, 0.0, 0.0));
	} else if (isUniform()) {
		fill(getUniformValue() * factor);
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		for (int c=0; c<num_arrays; ++c) {
			matty::getDevice(dev)->scale(getArray(dev, c), factor);
		}
		readUnlock(dev);
	}
}

void VectorMatrix::scale(const Vector3d &factors)
{
	if (factors == Vector3d(0.0, 0.0, 0.0)) {
		fill(Vector3d(0.0, 0.0, 0.0));
	} else if (isUniform()) {
		const Vector3d uni = getUniformValue();
		fill(Vector3d(uni.x*factors.x, uni.y*factors.y, uni.z*factors.z));
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		matty::getDevice(dev)->scale(getArray(dev, 0), factors.x);
		matty::getDevice(dev)->scale(getArray(dev, 1), factors.y);
		matty::getDevice(dev)->scale(getArray(dev, 2), factors.z);
		readUnlock(dev);
	}
}

void VectorMatrix::add(const VectorMatrix &op, double factor)
{
	if (this == &op) {
		scale(1.0 + factor);
	} else if (isUniform() && op.isUniform()) {
		fill(getUniformValue() + op.getUniformValue() * factor);
	} else {
		const int dev = computeStrategy2(op);
		writeLock(dev); op.readLock(dev);
		for (int c=0; c<num_arrays; ++c) {
			matty::getDevice(dev)->add(getArray(dev, c), op.getArray(dev, c), factor);
		}
		writeUnlock(dev); op.readUnlock(dev);
	}
}

Vector3d VectorMatrix::maximum() const
{
	if (isUniform()) {
		return getUniformValue();
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		const Vector3d max(
			matty::getDevice(dev)->maximum(getArray(dev, 0)),
			matty::getDevice(dev)->maximum(getArray(dev, 1)),
			matty::getDevice(dev)->maximum(getArray(dev, 2))
		);
		readUnlock(dev);
		return max;
	}
}

Vector3d VectorMatrix::sum() const
{
	if (isUniform()) {
		return getUniformValue() * getShape().getNumEl();
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		
		const Vector3d sum(
			matty::getDevice(dev)->sum(getArray(dev, 0)),
			matty::getDevice(dev)->sum(getArray(dev, 1)),
			matty::getDevice(dev)->sum(getArray(dev, 2))
		);
		readUnlock(dev);
		return sum;
	}
}

Vector3d VectorMatrix::average() const
{
	if (isUniform()) {
		return getUniformValue();
	} else {
		const int dev = computeStrategy1();
		readLock(dev);
		const Vector3d avg(
			matty::getDevice(dev)->average(getArray(dev, 0)),
			matty::getDevice(dev)->average(getArray(dev, 1)),
			matty::getDevice(dev)->average(getArray(dev, 2))
		);
		readUnlock(dev);
		return avg;
	}
}

void VectorMatrix::normalize(double len)
{
	if (isUniform()) {
		Vector3d uni = getUniformValue();
		uni.normalize(len);
		fill(uni);
	} else {
		const int dev = computeStrategy1();
		writeLock(dev); 
		getDevice(dev)->normalize3(getArray(dev, 0), getArray(dev, 1), getArray(dev, 2), len);
		writeUnlock(dev); 
	}
}

void VectorMatrix::normalize(const Matrix &len)
{
	if (isUniform() && len.isUniform()) {
		Vector3d uni = getUniformValue();
		uni.normalize(len.getUniformValue());
		fill(uni);
	} else {
		const int dev = computeStrategy2(len);
		writeLock(dev); len.readLock(dev);
		getDevice(dev)->normalize3(getArray(dev, 0), getArray(dev, 1), getArray(dev, 2), len.getArray(dev));
		len.readUnlock(dev); writeUnlock(dev); 
	}
}

void VectorMatrix::randomize()
{
	const int dev = computeStrategy1();
	writeLock(dev);
	for (int c=0; c<num_arrays; ++c) {
		matty::getDevice(dev)->randomize(getArray(dev, c));
	}
	writeUnlock(dev);
}

double VectorMatrix::absMax() const
{
	if (isUniform()) {
		const double x = uval[0], y = uval[1], z = uval[2];
		return std::sqrt(x*x+y*y+z*z);
	} else {
		const int dev = computeStrategy1();
		readLock(dev); 
		const double max = matty::getDevice(dev)->absmax3(
			this->getArray(dev, 0), this->getArray(dev, 1), this->getArray(dev, 2)
		);
		readUnlock(dev);
		return max;
	}
}

double VectorMatrix::dotSum(const VectorMatrix &other) const
{
	if (isUniform() && other.isUniform()) {
		const double x = uval[0], y = uval[1], z = uval[2];
		const double dot = x*x + y*y + z*z;
		return size() * dot;
	} else {
		const int dev = computeStrategy2(other);
		readLock(dev); if (this != &other) other.readLock(dev);
		const double sum = matty::getDevice(dev)->sumdot3(
			this->getArray(dev, 0), this->getArray(dev, 1), this->getArray(dev, 2),
			other.getArray(dev, 0), other.getArray(dev, 1), other.getArray(dev, 2)
		);
		if (this != &other) other.readUnlock(dev); readUnlock(dev);
		return sum;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream &operator<<(std::ostream &out, const VectorMatrix &mat)
{
	out << "VectorMatrix@" << (void*)&mat << std::endl;
	return out;
}

} // ns
