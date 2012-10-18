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
