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
#include "DeviceManager.h"

#include "cpu/CPUDevice.h"
#ifdef HAVE_CUDA
#include "cuda/CUDADevice.h"
#endif

#include <cassert>

namespace matty {

DeviceManager::DeviceManager()
{
}

DeviceManager::~DeviceManager()
{
}

int DeviceManager::addCPUDevice()
{
	devices.push_back(new CPUDevice());
	return devices.size()-1;
}

int DeviceManager::addCUDADevice(int cuda_device_id)
{
#ifdef HAVE_CUDA
	devices.push_back(new CU32Device(cuda_device_id));
#ifdef HAVE_CUDA_64
	devices.push_back(new CU64Device(cuda_device_id));
#endif
	return devices.size()-1;
#else
	assert(0); return -1;
#endif
}

} // ns
