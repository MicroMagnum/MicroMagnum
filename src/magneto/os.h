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
