#ifndef EXT_LINEAR_INTERPOLATE_H
#define EXT_LINEAR_INTERPOLATE_H

#include "config.h"
#include "matrix/matty.h"

namespace matty_ext
{
	VectorMatrix linearInterpolate(const VectorMatrix &src, Shape dest_dim);
	Matrix       linearInterpolate(const       Matrix &src, Shape dest_dim);
}

#endif

