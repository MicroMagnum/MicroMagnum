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
