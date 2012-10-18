#ifndef CPU_ARRAY_H
#define CPU_ARRAY_H

#include "config.h"
#include "restrict.h"

#include "../Array.h"

namespace matty {

class CPUDevice;

class CPUArray : public Array
{
public:
	CPUArray(const Shape &shape, CPUDevice *device);
	virtual ~CPUArray();

	virtual int getNumBytes() const;
	
	double *ptr() { return data; }
	const double *ptr() const { return data; }

private:
	double * RESTRICT data;

	friend class CPUDevice;
};

} // ns

#endif
