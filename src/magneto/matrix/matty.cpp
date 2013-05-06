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

#include "matty.h"

#include "config.h"

namespace matty
{
	static DeviceManager *the_devman = 0;

	void matty_initialize()
	{
		the_devman = new DeviceManager();

		the_devman->addCPUDevice();
	}

	void matty_deinitialize()
	{
		delete the_devman;
		the_devman = 0;
	}

	DeviceManager &getDeviceManager()
	{
		return *the_devman;
	}
}
