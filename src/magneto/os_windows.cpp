#include "config.h"
#include <windows.h>

#include "Logger.h"

namespace os {

std::string pathSeparator()
{
	return "\\";
}

double getTickCount()
{
	// NOTE: The GetSystemTimeAsFileTime resolution can be as low as 15ms!

	FILETIME ft;
	GetSystemTimeAsFileTime(&ft);

	unsigned __int64 tmp = 0; // unit: 1e-7 seconds
	tmp |= ft.dwHighDateTime;
	tmp <<= 32;
	tmp |= ft.dwLowDateTime;

	return static_cast<double>(tmp * 1e-4); // convert to ms
}

bool disable_SSE_for_FFTW()
{
#ifdef _WIN64
	return false;
#else
	// Disable SSE for Windows 32-bit builds.
	return true;
#endif
}

} // namespace os
