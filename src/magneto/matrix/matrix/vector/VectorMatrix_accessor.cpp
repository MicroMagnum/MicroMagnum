#include "config.h"
#include "VectorMatrix_accessor.h"
#include "VectorMatrix.h"

#include "device/cpu/CPUArray.h"

namespace matty {

VectorMatrixAccessor::VectorMatrixAccessor(VectorMatrix &mat) : mat(mat) 
{
	mat.writeLock(0); // 0 = CPUDevice!
	data_x = static_cast<CPUArray*>(mat.getArray(0, 0))->ptr();
	data_y = static_cast<CPUArray*>(mat.getArray(0, 1))->ptr();
	data_z = static_cast<CPUArray*>(mat.getArray(0, 2))->ptr();

	// Precalculate strides
	const int rank = mat.getShape().getRank();
	strides[0] = 1;
	strides[1] = strides[0] * (rank > 0 ? mat.getShape().getDim(0) : 1);
	strides[2] = strides[1] * (rank > 1 ? mat.getShape().getDim(1) : 1);
	strides[3] = strides[2] * (rank > 2 ? mat.getShape().getDim(2) : 1);
}

VectorMatrixAccessor::~VectorMatrixAccessor()
{
	mat.writeUnlock(0);
}

ConstVectorMatrixAccessor::ConstVectorMatrixAccessor(const VectorMatrix &mat) : mat(mat) 
{
	mat.readLock(0);
	data_x = static_cast<CPUArray*>(mat.getArray(0, 0))->ptr();
	data_y = static_cast<CPUArray*>(mat.getArray(0, 1))->ptr();
	data_z = static_cast<CPUArray*>(mat.getArray(0, 2))->ptr();

	// Precalculate strides
	const int rank = mat.getShape().getRank();
	strides[0] = 1;
	strides[1] = strides[0] * (rank > 0 ? mat.getShape().getDim(0) : 1);
	strides[2] = strides[1] * (rank > 1 ? mat.getShape().getDim(1) : 1);
	strides[3] = strides[2] * (rank > 2 ? mat.getShape().getDim(2) : 1);
}

ConstVectorMatrixAccessor::~ConstVectorMatrixAccessor()
{
	mat.readUnlock(0);
}

}; //ns
