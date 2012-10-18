#include "config.h"
#include "Array.h"

namespace matty {

Array::Array(const Shape &shape, Device *device)
	: shape(shape), device(device)
{
}

Array::~Array()
{
}

} // ns
