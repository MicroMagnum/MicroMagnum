#include "config.h"
#include "Magneto.h"
#include "gradient.h"
#include <stdexcept>
#include <cassert>

#ifdef HAVE_CUDA
#include "gradient_cuda.h"
#endif

void gradient_cpu(double delta_x, double delta_y, double delta_z, const double *phi, VectorMatrix &field)
{
	const int dim_x = field.dimX();
	const int dim_y = field.dimY();
	const int dim_z = field.dimZ();

	const int phi_sx = 1;
	const int phi_sy = (dim_x+1);
	const int phi_sz = (dim_x+1)*(dim_y+1);

	VectorMatrix::accessor field_acc(field);

	for (int z=0; z<dim_z; ++z)
	for (int y=0; y<dim_y; ++y)
	for (int x=0; x<dim_x; ++x) {
		const int i = phi_sx*x + phi_sy*y + phi_sz*z;

		const double dx = (+ phi[i+1*phi_sx+0*phi_sy+0*phi_sz] - phi[i+0*phi_sx+0*phi_sy+0*phi_sz]
		                   + phi[i+1*phi_sx+1*phi_sy+0*phi_sz] - phi[i+0*phi_sx+1*phi_sy+0*phi_sz]
		                   + phi[i+1*phi_sx+0*phi_sy+1*phi_sz] - phi[i+0*phi_sx+0*phi_sy+1*phi_sz]
		                   + phi[i+1*phi_sx+1*phi_sy+1*phi_sz] - phi[i+0*phi_sx+1*phi_sy+1*phi_sz]) / (4.0 * delta_x);

		const double dy = (+ phi[i+0*phi_sx+1*phi_sy+0*phi_sz] - phi[i+0*phi_sx+0*phi_sy+0*phi_sz]
		                   + phi[i+1*phi_sx+1*phi_sy+0*phi_sz] - phi[i+1*phi_sx+0*phi_sy+0*phi_sz]
		                   + phi[i+0*phi_sx+1*phi_sy+1*phi_sz] - phi[i+0*phi_sx+0*phi_sy+1*phi_sz]
		                   + phi[i+1*phi_sx+1*phi_sy+1*phi_sz] - phi[i+1*phi_sx+0*phi_sy+1*phi_sz]) / (4.0 * delta_y);

		const double dz = (+ phi[i+0*phi_sx+0*phi_sy+1*phi_sz] - phi[i+0*phi_sx+0*phi_sy+0*phi_sz]
		                   + phi[i+1*phi_sx+0*phi_sy+1*phi_sz] - phi[i+1*phi_sx+0*phi_sy+0*phi_sz]
		                   + phi[i+0*phi_sx+1*phi_sy+1*phi_sz] - phi[i+0*phi_sx+1*phi_sy+0*phi_sz]
		                   + phi[i+1*phi_sx+1*phi_sy+1*phi_sz] - phi[i+1*phi_sx+1*phi_sy+0*phi_sz]) / (4.0 * delta_z);

		field_acc.set(x, y, z, Vector3d(dx, dy, dz));
	}
}

void gradient(double delta_x, double delta_y, double delta_z, const Matrix &pot, VectorMatrix &field)
{
	const bool use_cuda = isCudaEnabled();

	if (use_cuda) {
#ifdef HAVE_CUDA
		Matrix::const_cu32_accessor pot_acc(pot);
		gradient_cuda(delta_x, delta_y, delta_z, pot_acc.ptr(), field);
#else
		assert(0);
#endif
	} else {
		Matrix::ro_accessor pot_acc(pot);
		gradient_cpu(delta_x, delta_y, delta_z, pot_acc.ptr(), field);
	}
}

