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
#include "runge_kutta_cpu.h"

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <cstddef>

void rk_prepare_step_cpu(
	int step,
	double h,
	ButcherTableau &tab,

	const VectorMatrix &k0,
	const VectorMatrix &k1,
	const VectorMatrix &k2,
	const VectorMatrix &k3,
	const VectorMatrix &k4,
	const VectorMatrix &k5,

	const VectorMatrix &y,
	VectorMatrix &ytmp)
{
	const size_t s = y.size();

	VectorMatrix::const_accessor y_acc(y); 
	VectorMatrix::const_accessor k0_acc(k0), k1_acc(k1), k2_acc(k2), k3_acc(k3), k4_acc(k4), k5_acc(k5);
	VectorMatrix::accessor ytmp_acc(ytmp);

	const double b10 = tab.b[1][0];
	const double b20 = tab.b[2][0], b21 = tab.b[2][1];
	const double b30 = tab.b[3][0], b31 = tab.b[3][1], b32 = tab.b[3][2];
	const double b40 = tab.b[4][0], b41 = tab.b[4][1], b42 = tab.b[4][2], b43 = tab.b[4][3];
	const double b50 = tab.b[5][0], b51 = tab.b[5][1], b52 = tab.b[5][2], b53 = tab.b[5][3], b54 = tab.b[5][4];

	switch (step) {
		case 1:
			for (size_t i=0; i<s; ++i) {
				ytmp_acc.set(i, y_acc.get(i) + h * (b10*k0_acc.get(i)));
			}
			break;

		case 2:
			for (size_t i=0; i<s; ++i) {
				ytmp_acc.set(i, 
					y_acc.get(i) + h * (  b20*k0_acc.get(i) 
				                            + b21*k1_acc.get(i))
				);
			}
			break;

		case 3: 
			for (size_t i=0; i<s; ++i) {
				ytmp_acc.set(i, 
					y_acc.get(i) + h * (  b30*k0_acc.get(i) 
				                            + b31*k1_acc.get(i) 
			                                    + b32*k2_acc.get(i)) 
				);
			}
			break;

		case 4:
			for (size_t i=0; i<s; ++i) {
				ytmp_acc.set(i, 
					y_acc.get(i) + h * (  b40*k0_acc.get(i) 
					                    + b41*k1_acc.get(i)
					                    + b42*k2_acc.get(i)
					                    + b43*k3_acc.get(i))
				);
			}
			break;

		case 5:
			for (size_t i=0; i<s; ++i) {
				ytmp_acc.set(i, 
					y_acc.get(i) + h * (  b50*k0_acc.get(i) 
					                    + b51*k1_acc.get(i)
					                    + b52*k2_acc.get(i)
					                    + b53*k3_acc.get(i)
					                    + b54*k4_acc.get(i))
				);
			}
			break;

		default:
			throw std::runtime_error("Cant handle runge-kutta methods with more than 6 steps (not implemented)");
	}
}

void rk_combine_result_cpu(
	double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2, const VectorMatrix &k3,
	VectorMatrix &y, VectorMatrix &y_error)
{
	const size_t s = y.size();
	VectorMatrix::accessor y_acc(y), y_error_acc(y_error);

	// tab
	const double  c0 = tab. c[0],  c1 = tab. c[1],  c2 = tab. c[2],  c3 = tab. c[3];
	const double ec0 = tab.ec[0], ec1 = tab.ec[1], ec2 = tab.ec[2], ec3 = tab.ec[3];

	VectorMatrix::const_accessor k0_acc(k0), k1_acc(k1), k2_acc(k2), k3_acc(k3);
	for (size_t i=0; i<s; ++i) {
		const Vector3d y_i = y_acc.get(i);
		y_acc.set(i, 
			y_i + h * (  c0*k0_acc.get(i) 
			           + c1*k1_acc.get(i)
				   + c2*k2_acc.get(i)
				   + c3*k3_acc.get(i))
		);
		y_error_acc.set(i,
			h * (  ec0*k0_acc.get(i) 
			     + ec1*k1_acc.get(i)
			     + ec2*k2_acc.get(i)
			     + ec3*k3_acc.get(i))
		);
	}
}

void rk_combine_result_cpu(
	const double h, ButcherTableau &tab,
	const VectorMatrix &k0, const VectorMatrix &k1, const VectorMatrix &k2,
	const VectorMatrix &k3, const VectorMatrix &k4, const VectorMatrix &k5,
	VectorMatrix &y, VectorMatrix &y_error)
{
	const size_t s = y.size();
	VectorMatrix::accessor y_acc(y), y_error_acc(y_error);

	// tab
	const double  c0 = tab. c[0],  c1 = tab. c[1],  c2 = tab. c[2],  c3 = tab. c[3],  c4 = tab. c[4],  c5 = tab. c[5];
	const double ec0 = tab.ec[0], ec1 = tab.ec[1], ec2 = tab.ec[2], ec3 = tab.ec[3], ec4 = tab.ec[4], ec5 = tab.ec[5];

	// Special case for c1==ec1==0 (as in RK45 and CC45 Butcher tableaus)
	if (c1 == 0 && ec1 == 0) {
		VectorMatrix::const_accessor k0_acc(k0), k2_acc(k2), k3_acc(k3), k4_acc(k4), k5_acc(k5);
		for (size_t i=0; i<s; ++i) {
			const Vector3d y_i = y_acc.get(i);
			y_acc.set(i, 
				y_i + h * (  c0*k0_acc.get(i) 
					   + c2*k2_acc.get(i) 
					   + c3*k3_acc.get(i) 
					   + c4*k4_acc.get(i) 
					   + c5*k5_acc.get(i))
			);
			y_error_acc.set(i,
				h * (  ec0*k0_acc.get(i) 
				     + ec2*k2_acc.get(i) 
				     + ec3*k3_acc.get(i) 
				     + ec4*k4_acc.get(i) 
				     + ec5*k5_acc.get(i))
			);
		}
	} else { // General case
		VectorMatrix::const_accessor k0_acc(k0), k1_acc(k1), k2_acc(k2), k3_acc(k3), k4_acc(k4), k5_acc(k5);
		for (size_t i=0; i<s; ++i) {
			const Vector3d y_i = y_acc.get(i);
			y_acc.set(i, 
				y_i + h * (  c0*k0_acc.get(i) 
				           + c1*k1_acc.get(i)
					   + c2*k2_acc.get(i) 
					   + c3*k3_acc.get(i) 
					   + c4*k4_acc.get(i) 
					   + c5*k5_acc.get(i))
			);
			y_error_acc.set(i,
				h * (  ec0*k0_acc.get(i) 
				     + ec1*k1_acc.get(i)
				     + ec2*k2_acc.get(i) 
				     + ec3*k3_acc.get(i) 
				     + ec4*k4_acc.get(i) 
				     + ec5*k5_acc.get(i))
			);
		}
	}
}

double rk_scaled_error_norm_cpu(double h, double eps_abs, double eps_rel, const VectorMatrix &y, const VectorMatrix &y_error)
{
	double norm;
	
	if (false) { // Maximum norm
		double max = 0.0;

		VectorMatrix::const_accessor y_acc(y), y_error_acc(y_error);

		const size_t s = y.size() * 3;
		for (size_t i=0; i<s; ++i) {
			const Vector3d value =       y_acc.get(i);
			const Vector3d error = y_error_acc.get(i);
			{
				const double D0 = eps_rel * std::fabs(value.x) + eps_abs;
				const double r  = std::fabs(error.x) / std::fabs(D0); // scaled error at equation i
				if (r > max) max = r;
			} {
				const double D0 = eps_rel * std::fabs(value.y) + eps_abs;
				const double r  = std::fabs(error.y) / std::fabs(D0); // scaled error at equation i
				if (r > max) max = r;
			} {
				const double D0 = eps_rel * std::fabs(value.z) + eps_abs;
				const double r  = std::fabs(error.z) / std::fabs(D0); // scaled error at equation i
				if (r > max) max = r;
			}
		}
		norm = max;

	} else { // Euclidian norm
		double sum = 0.0;

		VectorMatrix::const_accessor y_acc(y), y_error_acc(y_error);
		const size_t s = y.size();
		for (size_t i=0; i<s; ++i) {
			const Vector3d value =       y_acc.get(i);
			const Vector3d error = y_error_acc.get(i);

			{
				const double D0 = eps_rel * std::fabs(value.x) + eps_abs;
				const double r  = std::fabs(error.x) / std::fabs(D0); // scaled error at equation i
				sum += r*r;
			} {
				const double D0 = eps_rel * std::fabs(value.y) + eps_abs;
				const double r  = std::fabs(error.y) / std::fabs(D0); // scaled error at equation i
				sum += r*r;
			} {
				const double D0 = eps_rel * std::fabs(value.z) + eps_abs;
				const double r  = std::fabs(error.z) / std::fabs(D0); // scaled error at equation i
				sum += r*r;
			}
		}
		norm = std::sqrt(sum / (s*3));
	}

	return norm;
}
