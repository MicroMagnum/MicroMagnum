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
#include "fdm_slonchewski.h"

#include "mmm/constants.h"

void fdm_slonchewski(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	double a_j,
	const VectorMatrix &p, // spin polarization
	const Matrix &Ms,
	const Matrix &alpha,
	const VectorMatrix &M,
	VectorMatrix &dM)
{
	// Calculate: 
	//   c1*(M x (M x p)) + c2*(M x p)
	//
	//   c1(theta): damping factor
	//   c2(theta): precession factor
	//   Ms*cos(theta) = M*p

	Matrix::ro_accessor Ms_acc(Ms), alpha_acc(alpha);
	VectorMatrix::const_accessor p_acc(p);
	VectorMatrix::const_accessor M_acc(M);
	VectorMatrix::accessor dM_acc(dM);

	const int N = dim_x * dim_y * dim_z;
	for (int n=0; n<N; ++n) {
		const double alpha = alpha_acc.at(n);
		const double Ms    = Ms_acc.at(n);

		const Vector3d p = p_acc.get(n);
		const Vector3d M = M_acc.get(n);
		
		if (p == Vector3d(0.0, 0.0, 0.0)) continue;
		if (Ms == 0.0) continue;

		// Calculate precession and damping terms
		const Vector3d Mxp   = cross(M, p); // precession: u=mxp
		const Vector3d MxMxp = cross(M, Mxp); // damping:  t=mxu=mx(mxp)

		// add both terms to dm/dt in LLGE
		const double gamma_pr = GYROMAGNETIC_RATIO / (1.0 + alpha*alpha);

		Vector3d dM_n = dM_acc.get(n);
		dM_n.x += gamma_pr * a_j * (-MxMxp.x/Ms + Mxp.x*alpha);
		dM_n.y += gamma_pr * a_j * (-MxMxp.y/Ms + Mxp.y*alpha);
		dM_n.z += gamma_pr * a_j * (-MxMxp.z/Ms + Mxp.z*alpha);
		dM_acc.set(n, dM_n);
	}
}

/*
void fdm_slonchewski(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	double a_j,
	const VectorMatrix &p, // spin polarization
	const Matrix &Ms,
	const Matrix &alpha,
	VectorMatrix &dM)
{
	// Calculate: 
	//   c1*(M x (M x p)) + c2*(M x p)
	//
	//   c1(theta): damping factor
	//   c2(theta): precession factor
	//   Ms*cos(theta) = M*p

	Matrix::ro_accessor Ms_acc(Ms), alpha_acc(alpha);
	VectorMatrix::const_accessor p_acc(p);
	VectorMatrix::const_accessor M_acc(M);
	VectorMatrix::accessor dM_acc(dM);

	const double Lambda2 = params.Lambda * params.Lambda;

	const int N = model()->mesh->totalNodes();
	for (int n=0; n<N; ++n) {
		const double alpha = alpha_acc.at(n);
		const double Ms    = Ms_acc.at(n);

		const Vector3d p = vector_get(p_acc, n);
		const Vector3d M = vector_get(M_acc, n);
		
		if (p == Vector3d(0.0, 0.0, 0.0)) continue;
		if (Ms == 0.0) continue;

		double a_j;
		switch (mode) {
			case MODE_FIXED_AJ: {
				a_j = params.fixed_a_j;
				break;
			}

			case MODE_VARIABLE_AJ: {
				const double cos_theta = dot(M, p) / Ms;
				const double theta = std::acos(cos_theta);
				const double sin_theta = std::sin(theta);
				const double f1 = H_BAR / (2*MU0*ELECTRON_CHARGE) / params.thickness;
				const double f2 = params.P * Lambda2 * sin_theta / ((Lambda2+1)+(Lambda2-1)*cos_theta);
				a_j = f1*f2 * params.J / Ms;
			}

			default: assert(0);
		}

		// Calculate precession and damping terms
		const Vector3d u = cross(M, p); // precession: u=mxp
		const Vector3d t = cross(M, u); // damping:  t=mxu=mx(mxp)

		// add both terms to dm/dt in LLGE
		const double gamma_pr = GYROMAGNETIC_RATIO / (1.0 + alpha*alpha);

		Vector3d dM_n = vector_get(dM_acc, n);
		dM_n.x += gamma_pr * a_j * (-t.x/Ms + u.x*alpha);
		dM_n.y += gamma_pr * a_j * (-t.y/Ms + u.y*alpha);
		dM_n.z += gamma_pr * a_j * (-t.z/Ms + u.z*alpha);
		dM_acc.set(n, dM_n);
	}
}
*/
