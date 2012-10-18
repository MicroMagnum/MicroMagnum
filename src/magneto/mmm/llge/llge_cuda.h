#ifndef LLGE_CUDA_H
#define LLGE_CUDA_H

#include "config.h"
#include "matrix/matty.h"

// calculate: dM = f1*MxH + f2*Mx(MxH)
double llge_cu32(
	const Matrix &f1, const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM
);

#ifdef HAVE_CUDA_64
double llge_cu64(
	const Matrix &f1, const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM
);
#endif

#endif
