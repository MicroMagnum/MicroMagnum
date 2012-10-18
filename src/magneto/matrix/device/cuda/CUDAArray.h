#ifndef CUDA_ARRAY_H
#define CUDA_ARRAY_H

#include "config.h"

#include "../Array.h"

namespace matty {

class CU32Array : public Array
{
	friend class CU32Device;
public:
	CU32Array(const Shape &shape, class CU32Device *device);
	virtual ~CU32Array();

	float *ptr() { return data; }
	const float *ptr() const { return data; }
	int getNumBytes() const { return getShape().getNumEl() * sizeof(float); }

private:
	float *data;
	class CU32Device *cuda_device;
};

#ifdef HAVE_CUDA_64
class CU64Array : public Array
{
	friend class CU64Device;
public:
	CU64Array(const Shape &shape, class CU64Device *device);
	virtual ~CU64Array();

	double *ptr() { return data; }
	const double *ptr() const { return data; }
	int getNumBytes() const { return getShape().getNumEl() * sizeof(double); }

private:
	double *data;
	class CU64Device *cuda_device;
};
#endif

} // ns

#endif
