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

#ifndef ABSTRACT_MATRIX_H
#define ABSTRACT_MATRIX_H

#include "config.h"

#include <vector>
#include <algorithm>

#include "device/Device.h"
#include "device/Array.h"
#include "Shape.h"

namespace matty {

class AbstractMatrix
{
public:
	AbstractMatrix(const Shape &shape, int num_arrays);
	AbstractMatrix(const AbstractMatrix &other);
	virtual ~AbstractMatrix();
	void swap(AbstractMatrix &other);

	bool isUniform() const { return state == UNIFORM; }
	bool isUninitialized() const { return state == UNINITIALIZED; }
	bool isWriteLocked() const;
	bool isLocked() const;

	void markUninitialized();

	void readLock(int dev) const;
	void readUnlock(int dev) const;
	void writeLock(int dev);
	void writeUnlock(int dev);

	const Shape &getShape() const { return shape; }
	int dimX() const { return getShape().getDim(0); }
	int dimY() const { return getShape().getDim(1); }
	int dimZ() const { return getShape().getDim(2); }
	int size() const { return getShape().getNumEl(); }

	bool cache(int cache_dev) const;
	bool uncache(int uncache_dev) const;
	void flush() const;
	bool isCached(int dev) const;

	void inspect() const; // write debug info to cout

protected:
	int findCachedDevice(int prefered_dev = -1) const;
	void allocateArraysForDevice(int dev) const;
	void freeArraysForDevice(int dev) const;

	int computeStrategy1() const;
	int computeStrategy2(const AbstractMatrix &other) const;

	// DATA MEMBERS ////////////////////////////////////////////

	enum State
	{
		UNINITIALIZED,
		UNIFORM,
		DATA
	};

	struct DeviceInfo
	{
		DeviceInfo() : ro_locks(0), rw_locks(0), valid(false), arrays() {}
		DeviceInfo(const DeviceInfo &other) : ro_locks(0), rw_locks(0), valid(false), arrays() {}
		~DeviceInfo() {}

		void swap(DeviceInfo &other) {
			using std::swap;
			swap(ro_locks, other.ro_locks);
			swap(rw_locks, other.rw_locks);
			swap(valid, other.valid);
			swap(arrays, other.arrays);
		}

		int ro_locks, rw_locks;
		bool valid;
		std::vector<Array*> arrays;
	};

	// device info and storage
	mutable std::vector<DeviceInfo> info;
	// dimensions of this matrix
	Shape shape;
	// state
	State state;
	// number of components
	int num_arrays;
	// uniform component values (if state == UNIFORM). assert(uval.size() == num_arrays).
	std::vector<double> uval;
};

} // ns

#endif
