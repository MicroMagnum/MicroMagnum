#include "config.h"
#include "fdm_zhangli.h"
#include "fdm_zhangli_cpu.h"
#ifdef HAVE_CUDA
#include "fdm_zhangli_cuda.h"
#include <cuda_runtime.h>
#endif

#include "Magneto.h"
#include "Benchmark.h"

#include <cassert>

void fdm_zhangli(
	int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z, bool do_precess,
	const Matrix &P, const Matrix &xi, const Matrix &Ms, const Matrix &alpha,
	const VectorMatrix &j, const VectorMatrix &M,
	VectorMatrix &dM)
{
	const bool use_cuda = ::isCudaEnabled();
	if (!use_cuda) {
		TIC("spintorque");
		fdm_zhangli_cpu(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, do_precess, P, xi, Ms, alpha, j, M, dM);
		TOC("spintorque");
	} else {
#ifdef HAVE_CUDA
		CUTIC("spintorque");
		fdm_zhangli_cuda(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, do_precess, P, xi, Ms, alpha, j, M, dM, isCuda64Enabled());
		CUTOC("spintorque");
#else
		assert(0);
#endif
	}
}
