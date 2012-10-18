#ifndef ANISOTROPY_CUDA_H
#define ANISOTROPY_CUDA_H

#include "config.h"
#include "matrix/matty.h"

void uniaxial_anisotropy_cuda(
	const VectorMatrix &axis,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H,
	bool cuda64
);

void cubic_anisotropy_cuda(
	const VectorMatrix &axis1,
	const VectorMatrix &axis2,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H,
	bool cuda64
);

#endif
