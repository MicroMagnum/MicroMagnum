#ifndef ANISOTROPY_CPU_H
#define ANISOTROPY_CPU_H

#include "config.h"
#include "matrix/matty.h"

double uniaxial_anisotropy_cpu(
	const VectorMatrix &axis,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H
);

double cubic_anisotropy_cpu(
	const VectorMatrix &axis1,
	const VectorMatrix &axis2,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H
);

#endif
