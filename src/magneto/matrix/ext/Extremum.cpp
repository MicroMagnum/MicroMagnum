#include "config.h"
#include "Extremum.h"

#include <cmath>
#include <float.h> // FLT_MIN
#include <cassert>
#include <stdexcept>

namespace matty_ext {

static Vector3d findExtremum_cpu(VectorMatrix &M, int z_slice, int component);

Vector3d findExtremum(VectorMatrix &M, int z_slice, int component)
{
	return findExtremum_cpu(M, z_slice, component);
}

static double fit(double x0, double x1, double x2, double y0, double y1, double y2)
{
	const double x0_sq = x0*x0, x1_sq = x1*x1, x2_sq = x2*x2;
	return (-y0*x2_sq + y0*x1_sq + y1*x2_sq - y1*x0_sq - y2*x1_sq + y2*x0_sq) / (y0*x1 - y0*x2 - y1*x0 + y1*x2 - y2*x1 + y2*x0) / 2.0;
}

Vector3d findExtremum_cpu(VectorMatrix &M, int z_slice, int component)
{
	if (M.getShape().getRank() != 3) {
		throw std::runtime_error("findExtremum: Fixme: Need matrix of rank 3");
	}

	if (component < 0 || component > 2) {
		throw std::runtime_error("findExtremum: Invalid 'component' value, must be 0, 1 or 2.");
	}

	const int dim_x = M.getShape().getDim(0);
	const int dim_y = M.getShape().getDim(1);

	VectorMatrix::const_accessor M_acc(M);

	// Find cell with maximum absolute value
	double max_val = -1.0;
	int max_x = -1, max_y = -1;
	for (int y=1; y<dim_y-1; ++y)
	for (int x=1; x<dim_x-1; ++x) {
		const int val = std::fabs(M_acc.get(x, y, z_slice)[component]);
		if (val > max_val) {
			max_val = val;
			max_x = x;
			max_y = y;
		}
	}
	assert(max_x > 0);
	assert(max_y > 0);
	
	// Refine maximum by fitting to sub-cell precision
	const double xdir_vals[3] = {
		M_acc.get(max_x-1, max_y+0, z_slice)[component],
		M_acc.get(max_x+0, max_y+0, z_slice)[component],
		M_acc.get(max_x+1, max_y+0, z_slice)[component]
	};

	const double ydir_vals[3] = {
		M_acc.get(max_x+0, max_y-1, z_slice)[component],
		M_acc.get(max_x+0, max_y+0, z_slice)[component],
		M_acc.get(max_x+0, max_y+1, z_slice)[component]
	};

	return Vector3d(
		fit(max_x-1, max_x+0, max_x+1, xdir_vals[0], xdir_vals[1], xdir_vals[2]),
		fit(max_y-1, max_y+0, max_y+1, ydir_vals[0], ydir_vals[1], ydir_vals[2]),
		static_cast<double>(z_slice)
	);
}

} // ns
