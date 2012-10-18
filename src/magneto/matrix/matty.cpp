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
