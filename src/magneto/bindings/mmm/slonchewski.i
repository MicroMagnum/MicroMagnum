%{
#include "mmm/spintorque/fdm_slonchewski.h"
%}

void fdm_slonchewski(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	double a_j,
	const VectorMatrix &p, // spin polarization
	const Matrix &Ms,
	const Matrix &alpha,
	const VectorMatrix &M,
	VectorMatrix &dM
);
