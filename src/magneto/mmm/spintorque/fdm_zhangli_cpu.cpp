/*
 * Copyright 2012, 2013 by the Micromagnum authors.
 *
 * This file is part of MicroMagnum.
 * 
 * MicroMagnum is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * MicroMagnum is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "config.h"
#include "fdm_zhangli_cpu.h"

#include "mmm/constants.h"

static Vector3d zhangli_dMdt(
	int x, int y, int z,
	int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z,
	bool do_precess,
	double P, double xi, double Ms, double alpha,
	const Vector3d &j, VectorMatrix::const_accessor &M_acc
);

void fdm_zhangli_cpu(
	int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z,
	bool do_precess,
	double P, double xi, 
	const Matrix &Ms, const Matrix &alpha,
        const Vector3d &j, const VectorMatrix &M, 
	VectorMatrix &dM)
{
	VectorMatrix::const_accessor M_acc(M);
	Matrix::ro_accessor Ms_acc(Ms), alpha_acc(alpha);
	VectorMatrix::accessor dM_acc(dM);

	for (int z=0; z<dim_z; ++z)
	for (int y=0; y<dim_y; ++y)
	for (int x=0; x<dim_x; ++x) {	
		const int k = z*dim_x*dim_y + y*dim_x + x;

		const double Ms    = Ms_acc.at(k); 
		const double alpha = alpha_acc.at(k);

		dM_acc.set(k, zhangli_dMdt(x, y, z, dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, do_precess, P, xi, Ms, alpha, j, M_acc));
	}
}

void fdm_zhangli_cpu(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool do_precess,
	const Matrix &P, const Matrix &xi,
	const Matrix &Ms, const Matrix &alpha,
	const VectorMatrix &j, const VectorMatrix &M,
	VectorMatrix &dM)
{
	VectorMatrix::const_accessor M_acc(M), j_acc(j);
	Matrix::ro_accessor Ms_acc(Ms), alpha_acc(alpha), P_acc(P), xi_acc(xi);
	VectorMatrix::accessor dM_acc(dM);

	for (int z=0; z<dim_z; ++z)
	for (int y=0; y<dim_y; ++y)
	for (int x=0; x<dim_x; ++x) {	
		const int k = z*dim_x*dim_y + y*dim_x + x;

		const double Ms    = Ms_acc.at(k); 
		const double alpha = alpha_acc.at(k);
		const double P     = P_acc.at(k);
		const double xi    = xi_acc.at(k);
		const Vector3d j   = j_acc.get(k);

		dM_acc.set(k, zhangli_dMdt(x, y, z, dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, do_precess, P, xi, Ms, alpha, j, M_acc));
	}
}

// Helpers //////////////////////////////////////////////////////////////////////////////////////////////////

static void zhangli_factors(double P, double xi, double Ms, double alpha, double &f1, double &f2)
{
	const double b_j = P * MU_BOHR / (ELECTRON_CHARGE*Ms*(1.0+xi*xi));
	const double b_j_prime = b_j / (1.0 + alpha*alpha);

	f1 = -b_j_prime/(Ms*Ms)*(1.0+alpha*xi);
	f2 = -b_j_prime/(Ms)*(xi-alpha);
}

static void zhangli_factors_noprecess(double P, double xi, double Ms, double alpha, double &f1, double &f2)
{
	const double b_j = P * MU_BOHR / (ELECTRON_CHARGE*Ms*(1.0+xi*xi));
	const double b_j_prime = b_j / (1.0 + alpha*alpha);

	f1 = -b_j_prime/(Ms*Ms)*(alpha*xi);
	f2 = -b_j_prime/(Ms)*(-alpha);
}

static void zhangli_vectors(
	int x, int y, int z,
	int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z,
	VectorMatrix::const_accessor &M_acc,
	const Vector3d &j, // j is current at cell (x,y,z)
	Vector3d &A, Vector3d &B)
{
	const int dim_xy = dim_x * dim_y;
	const int k = z*dim_xy + y*dim_x + x;

	// calculate j_grad_M
	const Vector3d M = M_acc.get(k);

	Vector3d dM_dx;
	if (dim_x > 1) {
		const int l1 = k-1, l2 = k+1; 
		if (x == 0) {
			const Vector3d M_right = M_acc.get(l2);
			dM_dx = (M_right - M) / delta_x;
		} else if (x == dim_x-1) {
			const Vector3d M_left = M_acc.get(l1);
			dM_dx = (M - M_left) / delta_x;
		} else {
			const Vector3d M_left  = M_acc.get(l1);
			const Vector3d M_right = M_acc.get(l2);
			dM_dx = (M_right - M_left) / (2 * delta_x);
		}
	} else {
		dM_dx.assign(0.0, 0.0, 0.0);
	}

	Vector3d dM_dy;
	if (dim_y > 1) {
		const int l1 = k-dim_x, l2 = k+dim_x; 
		if (y == 0) {
			const Vector3d M_down = M_acc.get(l2);
			dM_dy = (M_down - M) / delta_y;
		} else if (y == dim_y-1) {
			const Vector3d M_up = M_acc.get(l1);
			dM_dy = (M - M_up) / delta_y;
		} else {
			const Vector3d M_up   = M_acc.get(l1);
			const Vector3d M_down = M_acc.get(l2);
			dM_dy = (M_down - M_up) / (2 * delta_y);
		}
	} else {
		dM_dy.assign(0.0, 0.0, 0.0);
	}

	Vector3d dM_dz;
	if (dim_z > 1) {
		const int l1 = k-dim_xy, l2 = k+dim_xy; 
		if (z == 0) {
			const Vector3d M_back = M_acc.get(l2);
			dM_dz = (M_back - M) / delta_z;
		} else if (z == dim_z-1) {
			const Vector3d M_forw  = M_acc.get(l1);
			dM_dz = (M - M_forw) / delta_z;
		} else {
			const Vector3d M_forw = M_acc.get(l1);
			const Vector3d M_back = M_acc.get(l2);
			dM_dz = (M_back - M_forw) / (2 * delta_z);
		}
	} else {
		dM_dz.assign(0.0, 0.0, 0.0);
	}

	const Vector3d j_grad_M(
		j.x*dM_dx.x + j.y*dM_dy.x + j.z*dM_dz.x,
		j.x*dM_dx.y + j.y*dM_dy.y + j.z*dM_dz.y,
		j.x*dM_dx.z + j.y*dM_dy.z + j.z*dM_dz.z
	);

	B = cross(M, j_grad_M);
	A = cross(M, B);
}

static Vector3d zhangli_dMdt(
	int x, int y, int z,
	int dim_x, int dim_y, int dim_z, double delta_x, double delta_y, double delta_z,
	bool do_precess,
	double P, double xi, double Ms, double alpha, 
	const Vector3d &j,
	VectorMatrix::const_accessor &M_acc)
{
	if (Ms == 0.0) return Vector3d(0.0, 0.0, 0.0);

	Vector3d A, B; double f1, f2;
	if (do_precess) {
		zhangli_factors(P, xi, Ms, alpha, f1, f2);
	} else {
		zhangli_factors_noprecess(P, xi, Ms, alpha, f1, f2);
	}
	zhangli_vectors(x, y, z, dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, M_acc, j, A, B);
	return f1*A + f2*B;
}
