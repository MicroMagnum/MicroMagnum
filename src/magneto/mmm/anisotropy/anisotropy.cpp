#include "config.h"
#include "anisotropy.h"
#include "anisotropy_cpu.h"
#ifdef HAVE_CUDA
#include "anisotropy_cuda.h"
#include <cuda_runtime.h>
#endif

#include "Magneto.h"
#include "Benchmark.h"

#include <cassert>

double uniaxial_anisotropy(
	const VectorMatrix &axis,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	const bool use_cuda = isCudaEnabled();

	double energy_sum = 0.0;

	if (use_cuda) {
#ifdef HAVE_CUDA
		CUTIC("uniaxial_anisotropy");
		uniaxial_anisotropy_cuda(axis, k, Ms, M, H, isCuda64Enabled());
		CUTOC("uniaxial_anisotropy");
#else
		assert(0);
#endif
	} else {
		TIC("uniaxial_anisotropy");
		energy_sum = uniaxial_anisotropy_cpu(axis, k, Ms, M, H);
		TOC("uniaxial_anisotropy");
	}

	return energy_sum;
}

double cubic_anisotropy(
	const VectorMatrix &axis1,
	const VectorMatrix &axis2,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	const bool use_cuda = isCudaEnabled();

	double energy_sum = 0.0;

	if (use_cuda) {
#ifdef HAVE_CUDA
		CUTIC("cubic_anisotropy");
		energy_sum = cubic_anisotropy_cuda(axis1, axis2, k, Ms, M, H, isCuda64Enabled());
		CUTOC("cubic_anisotropy");
#else
		assert(0);
#endif
	} else {
		TIC("cubic_anisotropy");
		energy_sum = cubic_anisotropy_cpu(axis1, axis2, k, Ms, M, H);
		TOC("cubic_anisotropy");
	}

	return energy_sum;
}
