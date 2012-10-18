#include "config.h"
#include "llge.h"

#include <cstddef>

void llge_cpu(
	const Matrix &f1, 
	const Matrix &f2,
	const VectorMatrix &M,
	const VectorMatrix &H,
	VectorMatrix &dM)
{
	VectorMatrix::accessor dM_acc(dM);
	VectorMatrix::const_accessor M_acc(M), H_acc(H);
	Matrix::ro_accessor f1_acc(f1), f2_acc(f2);

	// Calculate LLG: dM = -gamma'*(MxH) - (alpha*gamma'/Ms)*Mx(MxH)
	const size_t N = f1.size();
	for (size_t i=0; i<N; ++i) {
		const Vector3d     M = M_acc.get(i);
		const Vector3d     H = H_acc.get(i);
		const Vector3d   MxH = cross(M, H);
		const Vector3d MxMxH = cross(M, MxH);
		dM_acc.set(i, f1_acc.at(i)*MxH + f2_acc.at(i)*MxMxH);
	}
}
