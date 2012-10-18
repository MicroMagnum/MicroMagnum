#include "config.h"
#include "llge.h"
#include "llge_cpu.h"
#ifdef HAVE_CUDA
#include "llge_cuda.h"
#include <cuda_runtime.h>
#endif

#include "Magneto.h"
#include "Benchmark.h"

#include <cassert>

void llge(
	const Matrix &f1, const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM)
{
	const bool use_cuda = isCudaEnabled();

	if (use_cuda) {
#ifdef HAVE_CUDA
		CUTIC("llge");
#ifdef HAVE_CUDA_64
		if (isCuda64Enabled())
			llge_cu64(f1, f2, M, H, dM);
		else
#endif
			llge_cu32(f1, f2, M, H, dM);
		CUTOC("llge");
#else
		assert(0);
#endif
	} else {
		TIC("llge");
		llge_cpu(f1, f2, M, H, dM);
		TOC("llge");
	}
}
