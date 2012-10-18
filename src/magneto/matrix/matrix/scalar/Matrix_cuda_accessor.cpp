#include "config.h"
#include "Matrix_cuda_accessor.h"
#include "Matrix.h"

#include "device/cuda/CUDAArray.h"

#include "config.h"

namespace matty {

MatrixCU32Accessor::MatrixCU32Accessor(Matrix &mat) : mat(mat) 
{
	mat.writeLock(1); data = static_cast<CU32Array*>(mat.getArray(1))->ptr();
}

MatrixCU32Accessor::~MatrixCU32Accessor()
{
	mat.writeUnlock(1);
}

ConstMatrixCU32Accessor::ConstMatrixCU32Accessor(const Matrix &mat) : mat(mat) 
{
	mat.readLock(1); data = static_cast<const CU32Array*>(mat.getArray(1))->ptr();
}

ConstMatrixCU32Accessor::~ConstMatrixCU32Accessor()
{
	mat.readUnlock(1);
}

#ifdef HAVE_CUDA_64
MatrixCU64Accessor::MatrixCU64Accessor(Matrix &mat) : mat(mat) 
{
	mat.writeLock(2); data = static_cast<CU64Array*>(mat.getArray(2))->ptr();
}

MatrixCU64Accessor::~MatrixCU64Accessor()
{
	mat.writeUnlock(2);
}

ConstMatrixCU64Accessor::ConstMatrixCU64Accessor(const Matrix &mat) : mat(mat) 
{
	mat.readLock(2); data = static_cast<const CU64Array*>(mat.getArray(2))->ptr();
}

ConstMatrixCU64Accessor::~ConstMatrixCU64Accessor()
{
	mat.readUnlock(2);
}
#endif

} // ns
