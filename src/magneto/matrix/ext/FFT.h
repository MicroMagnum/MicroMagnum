#ifndef MATTY_EXT_FFT_H
#define MATTY_EXT_FFT_H

#include "config.h"
#include "matrix/matty.h"

#include <vector>

namespace matty_ext 
{
	void  fftn(ComplexMatrix &inout, const std::vector<int> &loop_dims_select = std::vector<int>());
	void ifftn(ComplexMatrix &inout, const std::vector<int> &loop_dims_select = std::vector<int>());
}

#endif
