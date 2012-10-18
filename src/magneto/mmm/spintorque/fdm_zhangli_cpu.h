#ifndef FDM_ZHANGLI_CPU_H
#define FDM_ZHANGLI_CPU_H

#include "config.h"
#include "fdm_zhangli.h"

void fdm_zhangli_cpu(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool do_precess,
	double P, double xi, 
	const Matrix &Ms, const Matrix &alpha,
        const Vector3d &j, const VectorMatrix &M, 
	VectorMatrix &dM
);

void fdm_zhangli_cpu(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool do_precess,
	const Matrix &P, const Matrix &xi,
	const Matrix &Ms, const Matrix &alpha,
	const VectorMatrix &j, const VectorMatrix &M,
	VectorMatrix &dM
);

#endif
