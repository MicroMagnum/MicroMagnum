#include "VectorVectorConvolution_Simple.h"
#include <assert>

VectorVectorConvolution_Simple::VectorVectorConvolution_Simple(const VectorMatrix &lhs, int dim_x, int dim_y, int dim_z)
	: lhs(lhs), dim_x(dim_x), dim_y(dim_y), dim_z(dim_z)
{
	assert(lhs.getShape().getDim(0) == 3);
	exp_x = lhs.getShape().getDim(1);
	exp_y = lhs.getShape().getDim(2);
	exp_z = lhs.getShape().getDim(3);
}

VectorVectorConvolution_Simple::~VectorVectorConvolution_Simple()
{
}

void VectorVectorConvolution_Simple::execute(const VectorMatrix &rhs, Matrix &res)
{
	VectorMatrix::const_accessor S_acc(lhs);
	VectorMatrix::const_accessor M_acc(rhs); 
	Matrix::      accessor phi_acc(res);

	// phi(r) = int S(r-r')*M(r') dr'
	// phi = Sx*Mx + Sy*My + Sz*Mz

	for (int z=0; z<dim_z; ++z)
	for (int y=0; y<dim_y; ++y)
	for (int x=0; x<dim_x; ++x) 
	{
		double sum = 0.0;

		for (int o=0; o<dim_z; ++o)
		for (int n=0; n<dim_y; ++n)
		for (int m=0; m<dim_x; ++m) 
		{
			// (X,Y,Z): position in demag tensor field matrix
			const int X = (x-m+exp_x) % exp_x;
			const int Y = (y-n+exp_y) % exp_y;
			const int Z = (z-o+exp_z) % exp_z;

			const Vector3d &S = S_acc.get(X, Y, Z);
			const Vector3d &M = M_acc.get(m, n, o);

			sum = S.x*M.x + S.y*M.y + S.z*M.z;
		}

		phi_acc.at(x, y, z) = sum;
	}
}
