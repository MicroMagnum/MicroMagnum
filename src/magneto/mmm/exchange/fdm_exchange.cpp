#include "config.h"
#include "fdm_exchange.h"
#include "fdm_exchange_cpu.h"
#ifdef HAVE_CUDA
#include "fdm_exchange_cuda.h"
#include <cuda_runtime.h>
#endif

#include "Magneto.h"
#include "Benchmark.h"

double fdm_exchange(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool periodic_x, bool periodic_y, bool periodic_z,
	const Matrix &Ms,
	const Matrix &A,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	const bool use_cuda = isCudaEnabled();

	double res = 0;
	if (use_cuda) {
#ifdef HAVE_CUDA
		CUTIC("exchange");
		res = fdm_exchange_cuda(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, periodic_x, periodic_y, periodic_z, Ms, A, M, H, isCuda64Enabled());
		CUTOC("exchange");
#else
		assert(0);
#endif
	} else {
		TIC("exchange");
		res = fdm_exchange_cpu(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, periodic_x, periodic_y, periodic_z, Ms, A, M, H);
		TOC("exchange");
	}

	return res;
}
