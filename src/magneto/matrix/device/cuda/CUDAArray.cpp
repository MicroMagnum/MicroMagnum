#include "config.h"
#include "CUDAArray.h"
#include "CUDADevice.h"

#include <stdexcept>

#include <cublas.h>
#include <cuda_runtime.h>

namespace matty {

CU32Array::CU32Array(const Shape &shape, CU32Device *device)
	: Array(shape, device), data(0), cuda_device(device)
{
	// Note: see also 'cudaMallocPitch', 'cudaMalloc3d'
	cudaError_t err = cudaMalloc((void**)&data, getShape().getNumEl() * sizeof(float));
	if (err != cudaSuccess) throw std::runtime_error("CU32Array::allocate: Could not allocate memory on GPU.");
}

CU32Array::~CU32Array()
{
	cudaFree(data);
}

#ifdef HAVE_CUDA_64
CU64Array::CU64Array(const Shape &shape, CU64Device *device)
	: Array(shape, device), data(0), cuda_device(device)
{
	// Note: see also 'cudaMallocPitch', 'cudaMalloc3d'
	cudaError_t err = cudaMalloc((void**)&data, getShape().getNumEl() * sizeof(double));
	if (err != cudaSuccess) throw std::runtime_error("CU64Array::allocate: Could not allocate memory on GPU.");
}

CU64Array::~CU64Array()
{
	cudaFree(data);
}
#endif

} // ns
