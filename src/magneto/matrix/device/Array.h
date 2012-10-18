#ifndef MATTY_ARRAY_H
#define MATTY_ARRAY_H

#include "config.h"
#include "Shape.h"

namespace matty {

class Device;

class Array
{
public:
	Array(const Shape &shape, Device *device);
	virtual ~Array();

	const Shape &getShape() const { return shape; }
	Device *getDevice() const { return device; }

	virtual int getNumBytes() const = 0;

private:
	Shape shape;
	Device *device;
};

} // ns

#endif
