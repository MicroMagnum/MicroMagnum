#ifndef LLGE_H
#define LLGE_H

#include "config.h"
#include "matrix/matty.h"

// calculate: dM = f1*MxH + f2*Mx(MxH)
void llge(
	const Matrix &f1, const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM
);

#endif
