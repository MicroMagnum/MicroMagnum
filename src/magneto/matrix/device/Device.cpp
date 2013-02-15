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
#include "Device.h"

#include "cpu/CPUDevice.h"

#include <cassert>
#include <ostream>

namespace matty {

Device::Device(const std::string &dev_name) : dev_name(dev_name), alloced_mem(0), alloced_arrays(0)
{
}

Device::~Device()
{
}

void Device::printReport(std::ostream &out)
{
	using namespace std;

	out << "Device " << dev_name.c_str() << ": ";
	out << alloced_mem << " bytes allocated in " << alloced_arrays << " arrays.";
	out << endl;
}

} // ns
