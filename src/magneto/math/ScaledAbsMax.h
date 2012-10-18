#ifndef SCALED_ABS_MAX_H
#define SCALED_ABS_MAX_H

#include "config.h"
#include "matrix/matty.h"

/*
 * Returns: M'[i] = max(abs(M[i]) / scale[i])
 */
double scaled_abs_max(VectorMatrix &M, Matrix &scale);

#endif
