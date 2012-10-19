#include "config.h"
#include "anisotropy_cpu.h"
#include "mmm/constants.h"

#include <cstddef>

double uniaxial_anisotropy_cpu(
	const VectorMatrix &axis,
	const Matrix &k,
	const Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	//
	// Calculate:
	//   H(i) = 2(i)k/(mu0*Ms^2) * (M(i)*axis) * axis
	//
	VectorMatrix::const_accessor M_acc(M);
	VectorMatrix::accessor H_acc(H);
	VectorMatrix::const_accessor axis_acc(axis);
	Matrix::ro_accessor Ms_acc(Ms), k_acc(k);

	double energy_sum = 0.0;

	// Compute field
	const size_t num_nodes = M.size();
	for (size_t i=0; i<num_nodes; ++i) {
		const double     Ms = Ms_acc.at(i);
		const double      k = k_acc.at(i);
		const Vector3d    M = M_acc.get(i);
		const Vector3d axis = axis_acc.get(i);

		if (Ms == 0.0) {
			H_acc.set(i, Vector3d(0.0, 0.0, 0.0));
		} else {
			const double d = dot(M, axis) / Ms;
			const Vector3d H = ((2.0 / MU0) * k * d / Ms) * axis;

			H_acc.set(i, H);
			//energy_sum += -k*d*d;
			energy_sum += k*(1.0 - d*d);
		}
	}

	return energy_sum;
}

double cubic_anisotropy_cpu(
	const VectorMatrix &axis1,
	const VectorMatrix &axis2,
	const Matrix &k,
	const Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	VectorMatrix::const_accessor M_acc(M);
	VectorMatrix::accessor H_acc(H);
	VectorMatrix::const_accessor axis1_acc(axis1);
	VectorMatrix::const_accessor axis2_acc(axis2);
	Matrix::ro_accessor Ms_acc(Ms), k_acc(k);

	// Compute field
	const size_t num_nodes = M.size();
	for (size_t i=0; i<num_nodes; ++i) {
		const double Ms = Ms_acc.at(i);
		if (Ms == 0.0) {
			H_acc.set(i, Vector3d(0.0, 0.0, 0.0));
		} else {
			const double k = k_acc.at(i);
			const Vector3d m = normalize(M_acc.get(i), 1.0);
			const Vector3d axis1 = axis1_acc.get(i);
			const Vector3d axis2 = axis2_acc.get(i);
			const Vector3d axis3 = cross(axis1, axis2);

			const double a1 = dot(axis1, m), a1sq = a1*a1;
			const double a2 = dot(axis2, m), a2sq = a2*a2;
			const double a3 = dot(axis3, m), a3sq = a3*a3;

			const Vector3d m1 = a1*axis1;
			const Vector3d m2 = a2*axis2;
			const Vector3d m3 = a3*axis3;
			const Vector3d H = (-2/MU0)*k/Ms * ((a2sq+a3sq)*m1 + (a1sq+a3sq)*m2 + (a1sq+a2sq)*m3);
			H_acc.set(i, H);

			//energy += k * (a1sq*a2sq+a1sq*a3sq+a2sq*a3sq);
		}
	}

	return 0.0;
}

/*
  for(UINT4m i=0;i<size;++i) {
    const ThreeVector& u1 = axis1[i];
    const ThreeVector& u2 = axis2[i];
    const ThreeVector&  m = spin[i];

    // This code requires u1 and u2 to be orthonormal, and m to be a
    // unit vector.  Basically, decompose
    //
    //             m = a1.u1 + a2.u2 + a3.u3
    //               =  m1   +  m2   +  m3
    //
    // where u3=u1xu2, a1=m*u1, a2=m*u2, a3=m*u3.
    //
    // Then the energy is
    //                 2  2     2  2     2  2
    //            K (a1 a2  + a1 a3  + a2 a3 )
    //
    // and the field in say the u1 direction is
    //              2    2                 2    2
    //         C (a2 + a3 ) a1 . u1 = C (a2 + a3 ) m1
    //
    // where C = -2K/(MU0 Ms).
    //
    // In particular, note that
    //           2         2     2
    //         a3  = 1 - a1  - a2
    // and
    //         m3  = m - m1 - m2
    //
    // This shows that energy and field can be computed without
    // explicitly calculating u3.  However, the cross product
    // evaluation to get u3 is not that expensive, and appears
    // to be more accurate.  At a minimum, in the above expressions
    // one should at least insure that a3^2 is non-negative.
*/

