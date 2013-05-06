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
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

#include "Logger.h"

namespace os {

std::string pathSeparator()
{
	return "/";
}

#if 1

double getTickCount()
{
	// This should be reasonably accurate: Time resolution is about 1e-6 sec on my Linux system.
	timeval ts;
	gettimeofday(&ts, 0);
	return ts.tv_sec * 1e3 + ts.tv_usec / 1e3;
}

#else 

// Linux: link with -lrt
double getTickCount()
{
	timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts); 
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

double getTickCountResolution()
{
	// this reports 1e-6 secs on my Linux system.
	timespec ts;
	clock_getres(CLOCK_REALTIME, &ts);
	return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#endif

bool disable_SSE_for_FFTW()
{
	return false;
}

} // namespace os
