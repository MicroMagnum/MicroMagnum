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

#ifndef MATRIX_LIB_H
#define MATRIX_LIB_H

#include "config.h"

#include "device/Device.h"
#include "device/DeviceManager.h"

#include "matrix/scalar/Matrix.h"
#include "matrix/vector/VectorMatrix.h"
#include "matrix/complex/ComplexMatrix.h"

#ifdef HAVE_CUDA
#include "device/cuda_tools.h"
#endif

namespace matty
{
	void matty_initialize();
	void matty_deinitialize();
	DeviceManager &getDeviceManager();
	inline Device *getDevice(int i) { return getDeviceManager().getDevice(i); }
}

using namespace matty;

#endif
