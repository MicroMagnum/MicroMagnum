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
