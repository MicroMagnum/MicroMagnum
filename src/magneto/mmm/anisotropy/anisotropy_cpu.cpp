#include "config.h"
#include "anisotropy_cpu.h"
#include "mmm/constants.h"

#include <cstddef>

// see comments at the end of the file
#define USE_CROSS_PRODUCT_LIKE_OOMMF 1

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

		if (Ms == 0.0 || k == 0.0) {
			H_acc.set(i, Vector3d(0.0, 0.0, 0.0));
		} else {
			const Vector3d axis = axis_acc.get(i);
			const Vector3d spin = M_acc.get(i) / Ms;

			const double d = dot(spin, axis);

			const Vector3d H = (2.0 * k * d / Ms / MU0) * axis;
			H_acc.set(i, H);

#ifdef USE_CROSS_PRODUCT_LIKE_OOMMF
			energy_sum += k * cross(axis, spin).abs_squared();
#else
			energy_sum += k * (1.0 - d*d);
#endif
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

	double energy_sum = 0.0;

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

			energy_sum += k * (a1sq*a2sq+a1sq*a3sq+a2sq*a3sq);
		}
	}

	return energy_sum;
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

    REAL8m k = K1[i];
    REAL8m field_mult = (-2/MU0)*k*Ms_inverse[i];
    if(field_mult==0.0) {
      energy[i]=0.0;
      field[i].Set(0.,0.,0.);
      continue;
    }

    ThreeVector u3 = u1;    u3 ^= u2;
    REAL8m a1 = u1*m;  REAL8m a1sq = a1*a1;
    REAL8m a2 = u2*m;  REAL8m a2sq = a2*a2;
    REAL8m a3 = u3*m;  REAL8m a3sq = a3*a3;

    energy[i] = k * (a1sq*a2sq+a1sq*a3sq+a2sq*a3sq);

    ThreeVector m1 = a1*u1;
    ThreeVector m2 = a2*u2;
    ThreeVector m3 = a3*u3;
    field[i]  = (a2sq+a3sq)*m1;
    field[i] += (a1sq+a3sq)*m2;
    field[i] += (a1sq+a2sq)*m3;
    field[i] *= field_mult;
  }
*/

/*
  for(UINT4m i=0;i<size;++i) {
    REAL8m k = K1[i];
    REAL8m field_mult = (2.0/MU0)*k*Ms_inverse[i];
    if(field_mult==0.0) {
      energy[i]=0.0;
      field[i].Set(0.,0.,0.);
      continue;
    }
    if(k<=0) {
      // Easy plane (hard axis)
      REAL8m dot = axis[i]*spin[i];
      field[i] = (field_mult*dot) * axis[i];
      energy[i] = -k*dot*dot; // Easy plane is zero energy
    } else {
      // Easy axis case.  For improved accuracy, we want to report
      // energy as -k*(dot*dot-1), where dot = axis * spin.  But
      // dot*dot-1 suffers from bad loss of precision if spin is
      // nearly parallel to axis.  The are a couple of ways around
      // this.  Recall that both spin and axis are unit vectors.
      // Then from the cross product:
      //            (axis x spin)^2 = 1 - dot*dot
      // The cross product requires 6 mults and 3 adds, and
      // the norm squared takes 3 mult and 2 adds
      //            => 9 mults + 5 adds.
      // Another option is to use
      //            (axis - spin)^2 = 2*(1-dot) 
      //     so  1 - dot*dot = t*(2-t)
      //                where t = 0.5*(axis-spin)^2.
      // The op count here is 
      //            => 5 mults + 6 adds.
      // Another advantage to the second approach is you get 'dot', as
      // opposed to dot*dot, which saves a sqrt if dot is needed.  The
      // downside is that if axis and spin are anti-parallel, then you
      // want to use (axis+spin)^2 rather than (axis-spin)^2.  I did
      // some single-spin test runs and the performance of the two
      // methods was about the same.  Below we use the cross-product
      // formulation. -mjd, 28-Jan-2001
      ThreeVector temp = axis[i];
      REAL8m dot = temp*spin[i];
      field[i] = (field_mult*dot) * temp;
      temp ^= spin[i];
      energy[i] = k*temp.MagSq();
    }
  }
*/
