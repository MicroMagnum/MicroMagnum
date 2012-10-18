#include "config.h"
#include "fdm_exchange_cpu.h"
#include "mmm/constants.h"

static double fdm_exchange_cpu_nonperiodic(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	const Matrix &Ms,
	const Matrix &A,
	const VectorMatrix &M,
	VectorMatrix &H
);

static double fdm_exchange_cpu_periodic(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool periodic_x, bool periodic_y, bool periodic_z,
	const Matrix &Ms,
	const Matrix &A,
	const VectorMatrix &M,
	VectorMatrix &H
);

double fdm_exchange_cpu(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool periodic_x, bool periodic_y, bool periodic_z,
	const Matrix &Ms,
	const Matrix &A,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	const bool periodic = periodic_x || periodic_y || periodic_z;
	if (periodic) {
		return fdm_exchange_cpu_periodic(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, periodic_x, periodic_y, periodic_z, Ms, A, M, H);
	} else {
		return fdm_exchange_cpu_nonperiodic(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, Ms, A, M, H);
	}
}

static double fdm_exchange_cpu_nonperiodic(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	const Matrix &Ms,
	const Matrix &A,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	const int dim_xy = dim_x * dim_y;
	const double wx = 1.0 / (delta_x * delta_x);
	const double wy = 1.0 / (delta_y * delta_y);
	const double wz = 1.0 / (delta_z * delta_z);

	VectorMatrix::const_accessor M_acc(M);
	VectorMatrix::accessor H_acc(H);
	Matrix::ro_accessor Ms_acc(Ms), A_acc(A);

	double energy = 0.0;
	for (int z=0; z<dim_z; ++z) {
		for (int y=0; y<dim_y; ++y) {	
			for (int x=0; x<dim_x; ++x) {
				const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)
				const double Ms = Ms_acc.at(i);
				if (Ms == 0.0) {
					H_acc.set(i, Vector3d(0.0, 0.0, 0.0));
					continue;
				}

				const int idx_l = i-     1;
				const int idx_r = i+     1;
				const int idx_u = i- dim_x;
				const int idx_d = i+ dim_x;
				const int idx_f = i-dim_xy;
				const int idx_b = i+dim_xy;

				const Vector3d M_i = M_acc.get(i) / Ms; // magnetization at (x,y,z)

				Vector3d sum(0.0, 0.0, 0.0);

				// left / right (X)
				if (x >       0) {
					const double Ms_l = Ms_acc.at(idx_l);
					if (Ms_l != 0.0) sum += ((M_acc.get(idx_l) / Ms_l) - M_i) * wx;
				}
				if (x < dim_x-1) {
					const double Ms_r = Ms_acc.at(idx_r);	
					if (Ms_r != 0.0) sum += ((M_acc.get(idx_r) / Ms_r) - M_i) * wx;
				}
				// up / down (Y)
				if (y >       0) {
					const double Ms_u = Ms_acc.at(idx_u);
					if (Ms_u != 0.0) sum += ((M_acc.get(idx_u) / Ms_u) - M_i) * wy;
				}
				if (y < dim_y-1) {
					const double Ms_d = Ms_acc.at(idx_d);
					if (Ms_d != 0.0) sum += ((M_acc.get(idx_d) / Ms_d) - M_i) * wy;
				}
				// forward / backward (Z)
				if (z >       0) {
					const double Ms_f = Ms_acc.at(idx_f);
					if (Ms_f != 0.0) sum += ((M_acc.get(idx_f) / Ms_f) - M_i) * wz;
				}
				if (z < dim_z-1) {
					const double Ms_b = Ms_acc.at(idx_b);
					if (Ms_b != 0.0) sum += ((M_acc.get(idx_b) / Ms_b) - M_i) * wz;
				}

				// Exchange field at (x,y,z)
				const Vector3d H_i = (2/MU0) * A_acc.at(i) * sum / Ms;
				H_acc.set(i, H_i);

				// Exchange energy sum
				energy += dot(M_i, H_i);
			}
		}
	}

	energy *= -MU0/2.0 * delta_x * delta_y * delta_z;
	return energy;
}

static double fdm_exchange_cpu_periodic(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool periodic_x, bool periodic_y, bool periodic_z,
	const Matrix &Ms,
	const Matrix &A,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	/*if (periodic_x && dim_x == 1) periodic_x = false;
	if (periodic_y && dim_y == 1) periodic_y = false;
	if (periodic_z && dim_z == 1) periodic_z = false;*/

	const int dim_xy = dim_x * dim_y;
	const double wx = 1.0 / (delta_x * delta_x);
	const double wy = 1.0 / (delta_y * delta_y);
	const double wz = 1.0 / (delta_z * delta_z);

	VectorMatrix::const_accessor M_acc(M);
	VectorMatrix::accessor H_acc(H);
	Matrix::ro_accessor Ms_acc(Ms), A_acc(A);

	double energy = 0.0;
	for (int z=0; z<dim_z; ++z) {
		for (int y=0; y<dim_y; ++y) {	
			for (int x=0; x<dim_x; ++x) {
				const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)
				const double Ms = Ms_acc.at(i);
				if (Ms == 0.0) {
					H_acc.set(i, Vector3d(0.0, 0.0, 0.0));
					continue;
				}

				int idx_l = i -      1;
				int idx_r = i +      1;
				int idx_u = i -  dim_x;
				int idx_d = i +  dim_x;
				int idx_f = i - dim_xy;
				int idx_b = i + dim_xy;

				// wrap-around for periodic boundary conditions
				if (periodic_x) {
					if (x ==       0) idx_l += dim_x;
					if (x == dim_x-1) idx_r -= dim_x;
				}
				if (periodic_y) {
					if (y ==       0) idx_u += dim_xy;
					if (y == dim_y-1) idx_d -= dim_xy;
				}
				if (periodic_z) {
					if (z ==       0) idx_f += dim_xy*dim_z;
					if (z == dim_z-1) idx_b -= dim_xy*dim_z;
				}

				const Vector3d M_i = M_acc.get(i) / Ms; // magnetization at (x,y,z)

				Vector3d sum(0.0, 0.0, 0.0);

				// left / right (X)
				if (x >       0 || periodic_x) {
					const double Ms_l = Ms_acc.at(idx_l);
					if (Ms_l != 0.0) sum += ((M_acc.get(idx_l) / Ms_l) - M_i) * wx;
				}
				if (x < dim_x-1 || periodic_x) {
					const double Ms_r = Ms_acc.at(idx_r);	
					if (Ms_r != 0.0) sum += ((M_acc.get(idx_r) / Ms_r) - M_i) * wx;
				}
				// up / down (Y)
				if (y >       0 || periodic_y) {
					const double Ms_u = Ms_acc.at(idx_u);
					if (Ms_u != 0.0) sum += ((M_acc.get(idx_u) / Ms_u) - M_i) * wy;
				}
				if (y < dim_y-1 || periodic_y) {
					const double Ms_d = Ms_acc.at(idx_d);
					if (Ms_d != 0.0) sum += ((M_acc.get(idx_d) / Ms_d) - M_i) * wy;
				}
				// forward / backward (Z)
				if (z >       0 || periodic_z) {
					const double Ms_f = Ms_acc.at(idx_f);
					if (Ms_f != 0.0) sum += ((M_acc.get(idx_f) / Ms_f) - M_i) * wz;
				}
				if (z < dim_z-1 || periodic_z) {
					const double Ms_b = Ms_acc.at(idx_b);
					if (Ms_b != 0.0) sum += ((M_acc.get(idx_b) / Ms_b) - M_i) * wz;
				}

				// Exchange field at (x,y,z)
				const Vector3d H_i = (2/MU0) * A_acc.at(i) * sum / Ms;
				H_acc.set(i, H_i);

				// Exchange energy sum
				energy += dot(M_i, H_i);
			}
		}
	}

	energy *= -MU0/2.0 * delta_x * delta_y * delta_z;
	return energy;
}
