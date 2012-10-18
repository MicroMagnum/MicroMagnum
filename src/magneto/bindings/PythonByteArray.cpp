#include "config.h"
#include "PythonByteArray.h"

#include <cstddef>

PythonByteArray::PythonByteArray()
	: length(0), arr(new char [0])
{
}

PythonByteArray::PythonByteArray(size_t length)
	: length(length), arr(new char [length])
{
}

PythonByteArray::~PythonByteArray()
{
}

char *PythonByteArray::get()
{
	return arr.ptr;
}

size_t PythonByteArray::getSize()
{
	return length;
}
