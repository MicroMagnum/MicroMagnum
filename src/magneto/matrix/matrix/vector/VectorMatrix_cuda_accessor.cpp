#include "config.h"
#include "VectorMatrix_cuda_accessor.h"
#include "VectorMatrix.h"

#include "device/cuda/CUDAArray.h"

namespace matty {

VectorMatrixCU32Accessor::VectorMatrixCU32Accessor(VectorMatrix &mat) : mat(mat) 
{
	mat.writeLock(1); // 1 = CU32Device!
	data_x = static_cast<CU32Array*>(mat.getArray(1, 0))->ptr();
	data_y = static_cast<CU32Array*>(mat.getArray(1, 1))->ptr();
	data_z = static_cast<CU32Array*>(mat.getArray(1, 2))->ptr();
}

VectorMatrixCU32Accessor::~VectorMatrixCU32Accessor()
{
	mat.writeUnlock(1);
}

ConstVectorMatrixCU32Accessor::ConstVectorMatrixCU32Accessor(const VectorMatrix &mat) : mat(mat) 
{
	mat.readLock(1);
	data_x = static_cast<CU32Array*>(mat.getArray(1, 0))->ptr();
	data_y = static_cast<CU32Array*>(mat.getArray(1, 1))->ptr();
	data_z = static_cast<CU32Array*>(mat.getArray(1, 2))->ptr();
}

ConstVectorMatrixCU32Accessor::~ConstVectorMatrixCU32Accessor()
{
	mat.readUnlock(1);
}

#ifdef HAVE_CUDA_64
VectorMatrixCU64Accessor::VectorMatrixCU64Accessor(VectorMatrix &mat) : mat(mat) 
{
	mat.writeLock(2); // 2 = CU64Device!
	data_x = static_cast<CU64Array*>(mat.getArray(2, 0))->ptr();
	data_y = static_cast<CU64Array*>(mat.getArray(2, 1))->ptr();
	data_z = static_cast<CU64Array*>(mat.getArray(2, 2))->ptr();
}

VectorMatrixCU64Accessor::~VectorMatrixCU64Accessor()
{
	mat.writeUnlock(2);
}

ConstVectorMatrixCU64Accessor::ConstVectorMatrixCU64Accessor(const VectorMatrix &mat) : mat(mat) 
{
	mat.readLock(2);
	data_x = static_cast<CU64Array*>(mat.getArray(2, 0))->ptr();
	data_y = static_cast<CU64Array*>(mat.getArray(2, 1))->ptr();
	data_z = static_cast<CU64Array*>(mat.getArray(2, 2))->ptr();
}

ConstVectorMatrixCU64Accessor::~ConstVectorMatrixCU64Accessor()
{
	mat.readUnlock(2);
}
#endif

} //ns
