#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include "config.h"
#include <vector>

namespace matty {

class Device;

class DeviceManager
{
public:
	DeviceManager();
	~DeviceManager();

	int addCPUDevice();
	int addCUDADevice(int cuda_device_id);

	Device *getDevice(int id) const { return devices[id]; }
	int getNumDevices() const { return devices.size(); }

private:
	std::vector<Device*> devices;
};

} // ns

#endif
