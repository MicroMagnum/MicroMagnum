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

#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <iostream>

#include "AbstractMatrix.h"
#include "matty.h"

namespace matty {

// Construction, destruction, copying, swapping //////////////////////////////////////////////////

AbstractMatrix::AbstractMatrix(const Shape &shape, int num_arrays)
	: shape(shape), state(AbstractMatrix::UNINITIALIZED), num_arrays(num_arrays)
{
	info.resize(matty::getDeviceManager().getNumDevices());
	for (size_t i=0; i<info.size(); ++i) info[i].arrays.resize(num_arrays, (Array*)0);
	
	uval.resize(num_arrays, 0.0);
}

AbstractMatrix::~AbstractMatrix()
{
	assert(!isLocked());

	for (size_t i=0; i<info.size(); ++i) {
		freeArraysForDevice(i);
	}
}

AbstractMatrix::AbstractMatrix(const AbstractMatrix &other)
	: shape(other.shape), state(other.state), num_arrays(other.num_arrays), uval(other.uval)
{
	info.resize(matty::getDeviceManager().getNumDevices());
	for (size_t i=0; i<info.size(); ++i) info[i].arrays.resize(num_arrays, (Array*)0);

	switch (state)
	{
		case UNINITIALIZED:
			// Nothing to do here...
			break;

		case UNIFORM:
			// Nothing to do here...
			break;

		case DATA: {
			if (other.isWriteLocked()) {
				throw std::runtime_error("Can't copy matrix: Source is write locked!");
			}
		
			// Copy up to one valid device array.
			const int dev = other.findCachedDevice();
			info[dev].valid = true;
			info[dev].ro_locks = 0;
			info[dev].rw_locks = 0;

			other.readLock(dev);
			Device * device = matty::getDeviceManager().getDevice(dev);
			for (int i=0; i<num_arrays; ++i) {
				info[dev].arrays[i] = device->makeArray(getShape());
				device->assign(this->info[dev].arrays[i], other.info[dev].arrays[i]);
			}
			other.readUnlock(dev);

			break;
		}
		default:
			assert(0);
	}
}

void AbstractMatrix::swap(AbstractMatrix &other)
{
	for (size_t i=0; i<info.size(); ++i) info[i].swap(other.info[i]);
	std::swap(shape, other.shape);
	std::swap(state, other.state);
	std::swap(uval, other.uval);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void AbstractMatrix::markUninitialized()
{
	if (isLocked()) {
		throw std::runtime_error("AbstractMatrix::markUninitialized: Cannot uninitialize because lock is in place.");
	}

	state = UNINITIALIZED;

	for (size_t i=0; i<info.size(); ++i) {
		info[i].valid = false;
	}
}

// Locking and unlocking //////////////////////////////////////////////////////////////////////////

// isWriteLocked, isLocked
// readLock, readUnlock
// writeLock, writeUnlock

bool AbstractMatrix::isWriteLocked() const
{
	for (size_t i=0; i<info.size(); ++i) {
		if (info[i].rw_locks > 0) return true;
	}
	return false;
}

bool AbstractMatrix::isLocked() const
{
	for (size_t i=0; i<info.size(); ++i) {
		if (info[i].ro_locks > 0) return true;
		if (info[i].rw_locks > 0) return true;
	}
	return false;
}

void AbstractMatrix::readLock(int dev) const
{
	// Make sure no other **write** locks are present
	if (isWriteLocked()) {
		throw std::runtime_error("Cannot acquire read lock while write locks are present.");
	}

	// Cache on device 'dev'
	if (!cache(dev)) {
		throw std::runtime_error("Failed to run cache() while aquiring read lock");
	}

	// Set lock
	info[dev].ro_locks += 1;
}

void AbstractMatrix::readUnlock(int dev) const
{
	if (!info[dev].valid) {
		throw std::runtime_error("Can't unlock because matrix lock is invalid!");
	}

	if (info[dev].ro_locks < 1 || info[dev].rw_locks != 0) {
		throw std::runtime_error("Can't unlock read lock because matrix is not read locked!");
	}

	info[dev].ro_locks -= 1;
}

void AbstractMatrix::writeLock(int dev) 
{
	// Make sure no other **read or write** locks are present
	if (isLocked()) {
		throw std::runtime_error("Can't lock because other lock is in place (1)");
	}

	// Get valid data on device 'dev'
	if (!cache(dev)) {
		throw std::runtime_error("Failed to run cache() during lock operation");
	}

	// Set lock
	info[dev].rw_locks += 1;

	// Invalidate data on all other devices
	for (size_t i=0; i<info.size(); ++i) {
		if (i != (size_t)dev) info[i].valid = false;
	}

	// Array can't be assumed to be uniform any more.
	state = DATA;
}

void AbstractMatrix::writeUnlock(int dev)
{
	if (!info[dev].valid) {
		throw std::runtime_error("Can't unlock because matrix lock is not valid!");
	}

	if (info[dev].ro_locks != 0 || info[dev].rw_locks != 1) {
		throw std::runtime_error("Can't unlock write lock because matrix is not write locked!");
	}

	// unlock	
	info[dev].rw_locks -= 1;
}

// Synchronizing data between devices //////////////////////////////////////////////////////////////////

// cache, uncache, flush
// isCached, findCachedDevice
// allocateArraysForDevice, freeArraysForDevice

bool AbstractMatrix::cache(int cache_dev) const
{
	// Already cached for device?
	if (info[cache_dev].valid) return true; // (1)

	// If not already cached: Can't cache if write lock exists.
	if (isWriteLocked()) return false;

	// Make sure array is allocated
	allocateArraysForDevice(cache_dev);

	switch (state) {
		case UNINITIALIZED: {
			// we don't care what is in the allocated memory block.
			break;
		}

		case UNIFORM: {
			// ..and fill with uniform value
			Device * device = matty::getDeviceManager().getDevice(cache_dev);
			for (int i=0; i<num_arrays; ++i) {
				device->fill(info[cache_dev].arrays[i], uval[i]);
			}
			break;
		}

		case DATA: {
			// Copy data from other device array (prefer CPU device)
			const int src_dev = findCachedDevice(0 /*CPU device prefered*/); // (2)

			// We need to copy the arrays from src_dev to cache_dev. 
			assert(src_dev != cache_dev); // because of (1) and (2)

			for (int i=0; i<num_arrays; ++i) {
				if (src_dev == 0) {
					Device * device = matty::getDeviceManager().getDevice(cache_dev);
					device->copyFromMemory(info[cache_dev].arrays[i] /*dst*/, (CPUArray*)info[src_dev].arrays[i] /*src*/);
				} else if (cache_dev == 0) {
					Device * device = matty::getDeviceManager().getDevice(src_dev);
					device->copyToMemory((CPUArray*)info[cache_dev].arrays[i] /*dst*/, info[src_dev].arrays[i] /*src*/);
				} else {
					using namespace std;
					cout << "src_dev = " << src_dev << ", dst_dev = " << cache_dev << endl;
					assert(0 && "Fixme: Not implemented!");
				}
			}
			break;
		}

		default: assert(0);
	}

	// Finally, mark array valid
	info[cache_dev].valid = true;
	info[cache_dev].rw_locks = 0;
	info[cache_dev].ro_locks = 0;
	return true;
}

bool AbstractMatrix::uncache(int uncache_dev) const
{
	// Device needs to be unlocked...
	if (info[uncache_dev].ro_locks > 0) return false;
	if (info[uncache_dev].rw_locks > 0) return false;

	// If data is held in memory blocks, make sure one other device array remains valid.
	if (state == DATA) {
		bool ok = false;
		for (size_t i=0; i<info.size(); ++i) {
			if (i != (size_t)uncache_dev && info[i].valid) ok = true;
		}
		if (!ok) return false;
	}

	// Ok. we can uncache.
	info[uncache_dev].valid = false;
	freeArraysForDevice(uncache_dev);

	return true;
}

void AbstractMatrix::flush() const
{
	switch (state) {
		case UNINITIALIZED:
		case UNIFORM:
			// Unload every device
			for (int i=0; i<matty::getDeviceManager().getNumDevices(); ++i) uncache(i);
			break;

		case DATA:
			// Synchronize with CPU
			if (!cache(0)) throw std::runtime_error("AbstractMatrix::flush: Failed to synchronize with CPU.");
			// Unload every other device
			for (int i=1; i<matty::getDeviceManager().getNumDevices(); ++i) uncache(i);
			break;

		default: assert(0);
	}
}

bool AbstractMatrix::isCached(int dev) const
{
	if (dev < 0 || dev >= int(info.size())) return false;
	return info[dev].valid;
}

int AbstractMatrix::findCachedDevice(int prefered_dev) const
{
	if (prefered_dev != -1 && info[prefered_dev].valid) {
		return prefered_dev;
	} else {
		int device_id = -1;
		for (size_t i=0; i<info.size(); ++i) {
			if (info[i].valid) {
				device_id = i;
			}
		}
		return device_id;
	}
}

void AbstractMatrix::allocateArraysForDevice(int dev) const
{
	Device * device = matty::getDeviceManager().getDevice(dev);
	for (int i=0; i<num_arrays; ++i) {
		if (info[dev].arrays[i] == 0) {	
			info[dev].arrays[i] = device->makeArray(getShape());
		}
	}
}

void AbstractMatrix::freeArraysForDevice(int dev) const
{
	for (int i=0; i<num_arrays; ++i) {
		if (info[dev].arrays[i] != 0) {	
			matty::getDevice(dev)->destroyArray(info[dev].arrays[i]);
			info[dev].arrays[i] = 0;
		}
	}
}

// Compute strategy //////////////////////////////////////////////////////////////////////////////// 

int AbstractMatrix::computeStrategy1() const
{
	int match_dev = -1; 

	for (size_t i=0; i<info.size(); ++i) {
		if (info[i].valid) match_dev = i;
	}

	if (match_dev == -1 && state != DATA) match_dev = 0;

	//cout << info.size() << " devices. selecting: " << match_dev << endl;
	return match_dev;
}

int AbstractMatrix::computeStrategy2(const AbstractMatrix &other) const
{
	int match_dev = -1; // full match
	int halfm_dev = -1; // half match

	for (size_t i=0; i<info.size(); ++i) {
		if (info[i].valid || other.info[i].valid) halfm_dev = i;
		if (info[i].valid && other.info[i].valid) match_dev = i;
	}

	if (match_dev == -1 && halfm_dev == -1 && state != DATA) match_dev = 0;

	return match_dev != -1 ? match_dev : halfm_dev;
}

// Debugging /////////////////////////////////////////////////////////////////////////////////////// 

void AbstractMatrix::inspect() const
{
	using namespace std;
	cout << "inspect:" << endl;
	cout << "  state = " << (   state == UNINITIALIZED ? "uninitialized"
	                          : state == DATA ? "data"
                                  : state == UNIFORM ? "uniform"
                                  : "<invalid>"
                                ) << endl;
	if (state == UNIFORM) {
		cout << "  uniform value = ";
		cout << uval[0]; for (int i=1; i<num_arrays; ++i) cout << ", " << uval[i];
		cout << endl;
	}

	for (size_t i=0; i<info.size(); ++i) {
		cout << "  device " << i << ":" << endl;
		cout << "    valid = " << info[i].valid << endl;
		cout << "    ro_locks = " << info[i].ro_locks << endl;
		cout << "    rw_locks = " << info[i].rw_locks << endl;
	} 
}

} // ns
