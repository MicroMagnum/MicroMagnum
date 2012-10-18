#include "config.h"
#include "Device.h"

#include "cpu/CPUDevice.h"

#include <cassert>
#include <ostream>

namespace matty {

Device::Device(const std::string &dev_name) : dev_name(dev_name), alloced_mem(0), alloced_arrays(0)
{
}

Device::~Device()
{
}

void Device::printReport(std::ostream &out)
{
	using namespace std;

	out << "Device " << dev_name.c_str() << ": ";
	out << alloced_mem << " bytes allocated in " << alloced_arrays << " arrays.";
	out << endl;
}

} // ns
