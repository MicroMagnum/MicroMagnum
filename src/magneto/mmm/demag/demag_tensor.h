#ifndef DEMAG_TENSOR_H
#define DEMAG_TENSOR_H

#include "config.h"
#include "matrix/matty.h"

// This function is exported to the Python code.
// Result: field of dimensions (6, exp_x, exp_y, exp_z)
Matrix GenerateDemagTensor(
	int dim_x, int dim_y, int dim_z, 
	double delta_x, double delta_y, double delta_z, 
	bool periodic_x, bool periodic_y, bool periodic_z, int periodic_repeat,
	int padding,
	const char *cache_dir
);

#endif
