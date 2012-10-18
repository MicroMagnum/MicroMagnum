#include "config.h"
#include "CPUArray.h"
#include "CPUDevice.h"

namespace matty {

CPUArray::CPUArray(const Shape &shape, CPUDevice *device)
	: Array(shape, device), data(0)
{
	data = new double [getShape().getNumEl()];
}

CPUArray::~CPUArray()
{
	delete [] data;
}

int CPUArray::getNumBytes() const
{
	return getShape().getNumEl() * sizeof(double);
}

} // ns
