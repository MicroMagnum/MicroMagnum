#ifndef EXT_EXTREMUM_H
#define EXT_EXTREMUM_H

#include "config.h"
#include "matrix/matty.h"

namespace matty_ext 
{
	Vector3d findExtremum(VectorMatrix &M, int z_slice, int component /*0, 1 or 2*/);
}

#endif
