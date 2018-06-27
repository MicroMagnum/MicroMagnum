#ifndef MAGNUM_CONFIG_H
#define MAGNUM_CONFIG_H

#if defined(_WIN32) || defined(_WIN64)
// On Windows, we don't use CMake to generate a config file, so we define
// our stuff manually.

#pragma warning( disable : 4003 4267 4996 4244)  // Disable warning messages

#else
#include "config.cmake-generated.h"
#endif

#endif
