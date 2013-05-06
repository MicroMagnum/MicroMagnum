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

#ifndef OS_H_INCLUDED
#define OS_H_INCLUDED

#include "config.h"
#include <string>

namespace os
{
	std::string pathSeparator();

	double getTickCount(); // one tick is one millisecond

	// FFTW crashes on 32 bit Windows when SSE is enabled.
	// see http://fftw.org/install/windows.html
	bool disable_SSE_for_FFTW();
}

#endif
