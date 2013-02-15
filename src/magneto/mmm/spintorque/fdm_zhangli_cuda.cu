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
#include "fdm_zhangli_cuda.h"
#include "mmm/constants.h"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Cross product: A = B x C
template <typename real>
static __inline__ __device__ void gpu_cross(
	real *ax, real *ay, real *az,
	real  bx, real  by, real  bz,
	real  cx, real  cy, real  cz)
{
	*ax = by*cz - bz*cy;
	*ay = bz*cx - bx*cz;
	*az = bx*cy - by*cx;
}

///////////////////////////////////////////// Gradient of magnetization (2d and 3d) /////////////////////////////////////////////

// 3D:
//                 Mx       grad Mx       dM_dx_x  dM_dx_x  dM_dx_x 
// grad M = grad ( My ) = ( grad My ) = ( dM_dy_y  dM_dy_y  dM_dy_y )
//                 Mz       grad Mz       dM_dz_z  dM_dz_z  dM_dz_z 

template <typename real>
static __inline__ __device__ void calculate_grad_M_2d(
	int dim_x, int dim_y, real delta_x, real delta_y,
	int x, int y,
	const real *mag_x, const real *mag_y, const real *mag_z,

	/*out 1*/
	real *dM_dx_x, real *dM_dx_y, real *dM_dx_z, // dM/dx vector
	real *dM_dy_x, real *dM_dy_y, real *dM_dy_z,

	/* out 2*/
	real *M_k_x_, real *M_k_y_, real *M_k_z_   // mag vector at (x,y)
)
{
	const int k = y*dim_x + x; // k: linear index of (x,y)
	const real M_k_x = mag_x[k], M_k_y = mag_y[k], M_k_z = mag_z[k];
	
	// out 2:
	*M_k_x_ = M_k_x;
	*M_k_y_ = M_k_y;
	*M_k_z_ = M_k_z;

	// out 1: calculate grad_M at position k
	*dM_dx_x = 0.0; *dM_dx_y = 0.0; *dM_dx_z = 0.0;
	*dM_dy_x = 0.0; *dM_dy_y = 0.0; *dM_dy_z = 0.0;

	if (dim_x > 1) {
		const int l1 = k-1, l2 = k+1; 
		if (x == 0) {
			const real M_right_x = mag_x[l2], M_right_y = mag_y[l2], M_right_z = mag_z[l2];
			*dM_dx_x = (M_right_x - M_k_x) / delta_x;
			*dM_dx_y = (M_right_y - M_k_y) / delta_x;
			*dM_dx_z = (M_right_z - M_k_z) / delta_x;
		} else if (x == dim_x-1) {
			const real M_left_x = mag_x[l1], M_left_y = mag_y[l1], M_left_z = mag_z[l1];
			*dM_dx_x = (M_k_x - M_left_x) / delta_x;
			*dM_dx_y = (M_k_y - M_left_y) / delta_x;
			*dM_dx_z = (M_k_z - M_left_z) / delta_x;
		} else {
			const real  M_left_x = mag_x[l1],  M_left_y = mag_y[l1],  M_left_z = mag_z[l1];
			const real M_right_x = mag_x[l2], M_right_y = mag_y[l2], M_right_z = mag_z[l2];
			*dM_dx_x = (M_right_x - M_left_x) / (2 * delta_x);
			*dM_dx_y = (M_right_y - M_left_y) / (2 * delta_x);
			*dM_dx_z = (M_right_z - M_left_z) / (2 * delta_x);
		}
	}

	if (dim_y > 1) {
		const int l1 = k-dim_x, l2 = k+dim_x; 
		if (y == 0) {
			const real M_down_x = mag_x[l2], M_down_y = mag_y[l2], M_down_z = mag_z[l2];
			*dM_dy_x = (M_down_x - M_k_x) / delta_y;
			*dM_dy_y = (M_down_y - M_k_y) / delta_y;
			*dM_dy_z = (M_down_z - M_k_z) / delta_y;
		} else if (y == dim_y-1) {
			const real M_up_x = mag_x[l1], M_up_y = mag_y[l1], M_up_z = mag_z[l1];
			*dM_dy_x = (M_k_x - M_up_x) / delta_y;
			*dM_dy_y = (M_k_y - M_up_y) / delta_y;
			*dM_dy_z = (M_k_z - M_up_z) / delta_y;
		} else {
			const real M_up_x   = mag_x[l1],   M_up_y = mag_y[l1],   M_up_z = mag_z[l1];
			const real M_down_x = mag_x[l2], M_down_y = mag_y[l2], M_down_z = mag_z[l2];
			*dM_dy_x = (M_down_x - M_up_x) / (2 * delta_y);
			*dM_dy_y = (M_down_y - M_up_y) / (2 * delta_y);
			*dM_dy_z = (M_down_z - M_up_z) / (2 * delta_y);
		}
	}
}

template <typename real>
static __inline__ __device__ void calculate_grad_M_3d(
	int dim_x, int dim_y, int dim_z, real delta_x, real delta_y, real delta_z,
	int x, int y, int z,
	const real *mag_x, const real *mag_y, const real *mag_z,

	/*out 1*/
	real *dM_dx_x, real *dM_dx_y, real *dM_dx_z, // dM/dx vector
	real *dM_dy_x, real *dM_dy_y, real *dM_dy_z,
	real *dM_dz_x, real *dM_dz_y, real *dM_dz_z,

	/* out 2*/
	real *M_k_x_, real *M_k_y_, real *M_k_z_ // mag vector at (x,y)
)
{
	const int dim_xy = dim_x * dim_y;
	const int k = z*dim_xy + y*dim_x + x; // k: linear index of (x,y)
	const real M_k_x = mag_x[k], M_k_y = mag_y[k], M_k_z = mag_z[k];
	
	// out 2:
	*M_k_x_ = M_k_x;
	*M_k_y_ = M_k_y;
	*M_k_z_ = M_k_z;

	// out 1: calculate grad_M at position k
	*dM_dx_x = 0.0; *dM_dx_y = 0.0; *dM_dx_z = 0.0;
	*dM_dy_x = 0.0; *dM_dy_y = 0.0; *dM_dy_z = 0.0;
	*dM_dz_x = 0.0; *dM_dz_y = 0.0; *dM_dz_z = 0.0;

	if (dim_x > 1) {
		const int l1 = k-1, l2 = k+1; 
		if (x == 0) {
			const real M_right_x = mag_x[l2], M_right_y = mag_y[l2], M_right_z = mag_z[l2];
			*dM_dx_x = (M_right_x - M_k_x) / delta_x;
			*dM_dx_y = (M_right_y - M_k_y) / delta_x;
			*dM_dx_z = (M_right_z - M_k_z) / delta_x;
		} else if (x == dim_x-1) {
			const real M_left_x = mag_x[l1], M_left_y = mag_y[l1], M_left_z = mag_z[l1];
			*dM_dx_x = (M_k_x - M_left_x) / delta_x;
			*dM_dx_y = (M_k_y - M_left_y) / delta_x;
			*dM_dx_z = (M_k_z - M_left_z) / delta_x;
		} else {
			const real  M_left_x = mag_x[l1],  M_left_y = mag_y[l1],  M_left_z = mag_z[l1];
			const real M_right_x = mag_x[l2], M_right_y = mag_y[l2], M_right_z = mag_z[l2];
			*dM_dx_x = (M_right_x - M_left_x) / (2 * delta_x);
			*dM_dx_y = (M_right_y - M_left_y) / (2 * delta_x);
			*dM_dx_z = (M_right_z - M_left_z) / (2 * delta_x);
		}
	}

	if (dim_y > 1) {
		const int l1 = k-dim_x, l2 = k+dim_x; 
		if (y == 0) {
			const real M_down_x = mag_x[l2], M_down_y = mag_y[l2], M_down_z = mag_z[l2];
			*dM_dy_x = (M_down_x - M_k_x) / delta_y;
			*dM_dy_y = (M_down_y - M_k_y) / delta_y;
			*dM_dy_z = (M_down_z - M_k_z) / delta_y;
		} else if (y == dim_y-1) {
			const real M_up_x = mag_x[l1], M_up_y = mag_y[l1], M_up_z = mag_z[l1];
			*dM_dy_x = (M_k_x - M_up_x) / delta_y;
			*dM_dy_y = (M_k_y - M_up_y) / delta_y;
			*dM_dy_z = (M_k_z - M_up_z) / delta_y;
		} else {
			const real M_up_x   = mag_x[l1],   M_up_y = mag_y[l1],   M_up_z = mag_z[l1];
			const real M_down_x = mag_x[l2], M_down_y = mag_y[l2], M_down_z = mag_z[l2];
			*dM_dy_x = (M_down_x - M_up_x) / (2 * delta_y);
			*dM_dy_y = (M_down_y - M_up_y) / (2 * delta_y);
			*dM_dy_z = (M_down_z - M_up_z) / (2 * delta_y);
		}
	}

	if (dim_z > 1) {
		const int l1 = k-dim_xy, l2 = k+dim_xy;
		if (z == 0) {
			const real M_back_x = mag_x[l2], M_back_y = mag_y[l2], M_back_z = mag_z[l2];
			*dM_dz_x = (M_back_x - M_k_x) / delta_z;
			*dM_dz_y = (M_back_y - M_k_y) / delta_z;
			*dM_dz_z = (M_back_z - M_k_z) / delta_z;
		} else if (z == dim_z-1) {
			const real M_front_x = mag_x[l1], M_front_y = mag_y[l1], M_front_z = mag_z[l1];
			*dM_dz_x = (M_k_x - M_front_x) / delta_z;
			*dM_dz_y = (M_k_y - M_front_y) / delta_z;
			*dM_dz_z = (M_k_z - M_front_z) / delta_z;
		} else {
			const real M_front_x = mag_x[l1], M_front_y = mag_y[l1], M_front_z = mag_z[l1];
			const real M_back_x  = mag_x[l2], M_back_y  = mag_y[l2],  M_back_z = mag_z[l2];
			*dM_dz_x = (M_back_x - M_front_x) / (2 * delta_z);
			*dM_dz_y = (M_back_y - M_front_y) / (2 * delta_z);
			*dM_dz_z = (M_back_z - M_front_z) / (2 * delta_z);
		}
	}
}

/////////////////////////////////////////////// Spin torque kernels /////////////////////////////////////////////////

template <typename real, bool do_precess>
__global__ 
static void kernel_zhangli_2d_naive(
	const real *Mx, const real *My, const real *Mz, 
	const real *Ms,
	const real *alpha,
	real *dmdt_x, real *dmdt_y, real *dmdt_z,
	const real *P, const real *xi, const real *jx, const real *jy, const real *jz,
	int dim_x, int dim_y, real delta_x, real delta_y)
{
	// Cell index
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < dim_x && y < dim_y) {
		const int k = y*dim_x + x; // k: linear index of (x,y)
		const real jx_k    = jx[k], jy_k = jy[k]; //, jz_k = jz[k];
		const real Ms_k    = Ms[k];
		const real alpha_k = alpha[k];
		const real P_k     = P[k];
		const real xi_k    = xi[k];

		real Mx_k, My_k, Mz_k;
		real dM_dx_x, dM_dx_y, dM_dx_z;
		real dM_dy_x, dM_dy_y, dM_dy_z;

		calculate_grad_M_2d(
			dim_x, dim_y, delta_x, delta_y, x, y, Mx, My, Mz,
			&dM_dx_x, &dM_dx_y, &dM_dx_z,
			&dM_dy_x, &dM_dy_y, &dM_dy_z,
			&Mx_k, &My_k, &Mz_k
		);

		// calculate (j*grad) M
		const real j_grad_M_x = jx_k*dM_dx_x + jy_k*dM_dy_x;
		const real j_grad_M_y = jx_k*dM_dx_y + jy_k*dM_dy_y;
		const real j_grad_M_z = jx_k*dM_dx_z + jy_k*dM_dy_z;

		// c = M x j_grad_M
		real cx, cy, cz;
		gpu_cross(&cx, &cy, &cz, Mx_k, My_k, Mz_k, j_grad_M_x, j_grad_M_y, j_grad_M_z);

		// d = M x c = M x (M x j_grad_M)
		real dx, dy, dz;
		gpu_cross(&dx, &dy, &dz, Mx_k, My_k, Mz_k, cx, cy, cz);

		if (Ms_k > 0.0) {
			// b_j
			const real b_j = P_k * MU_BOHR / (ELECTRON_CHARGE*Ms_k*(1.0+xi_k*xi_k));
			const real b_j_prime = b_j / (1.0 + alpha_k*alpha_k);
			
			// motion and distortion factors
			real motion_factor, distortion_factor;
			if (do_precess) {
				motion_factor     = -b_j_prime/(Ms_k*Ms_k)*(1.0+alpha_k*xi_k);
				distortion_factor = -b_j_prime/(Ms_k)*(xi_k-alpha_k);
			} else {
				motion_factor     = -b_j_prime/(Ms_k*Ms_k)*(alpha_k*xi_k);
				distortion_factor = -b_j_prime/(Ms_k)*(-alpha_k);
			}

			// add to llg terms
			dmdt_x[k] = motion_factor*dx + distortion_factor*cx;
			dmdt_y[k] = motion_factor*dy + distortion_factor*cy;
			dmdt_z[k] = motion_factor*dz + distortion_factor*cz;
		} else {
			dmdt_x[k] = 0.0;
			dmdt_y[k] = 0.0;
			dmdt_z[k] = 0.0;
		}
	}
}

template <typename real, bool do_precess>
__global__ 
static void kernel_zhangli_3d_naive(
	const real *Mx, const real *My, const real *Mz, 
	const real *Ms,
	const real *alpha,
	real *dmdt_x, real *dmdt_y, real *dmdt_z, 
	const real *P, const real *xi, const real *jx, const real *jy, const real *jz,
	int dim_x, int dim_y, int dim_z, real delta_x, real delta_y, real delta_z)
{
	// Cell index
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int dim_xy = dim_x * dim_y;

	for (int z=0; z<dim_z; ++z) {
		if (x < dim_x && y < dim_y) {
			const int k = z*dim_xy + y*dim_x + x; // k: linear index of (x,y)
			const real jx_k = jx[k], jy_k = jy[k], jz_k = jz[k];
			const real Ms_k    = Ms[k];
			const real alpha_k = alpha[k];
			const real P_k     = P[k];
			const real xi_k    = xi[k];

			real Mx_k, My_k, Mz_k;
			real dM_dx_x, dM_dx_y, dM_dx_z;
			real dM_dy_x, dM_dy_y, dM_dy_z;
			real dM_dz_x, dM_dz_y, dM_dz_z;

			calculate_grad_M_3d<real>(
				dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, x, y, z, Mx, My, Mz,
				&dM_dx_x, &dM_dx_y, &dM_dx_z,
				&dM_dy_x, &dM_dy_y, &dM_dy_z,
				&dM_dz_x, &dM_dz_y, &dM_dz_z,
				&Mx_k, &My_k, &Mz_k
			);

			// calculate (j*grad) M
			const real j_grad_M_x = jx_k*dM_dx_x + jy_k*dM_dy_x + jz_k*dM_dz_x;
			const real j_grad_M_y = jx_k*dM_dx_y + jy_k*dM_dy_y + jz_k*dM_dz_y;
			const real j_grad_M_z = jx_k*dM_dx_z + jy_k*dM_dy_z + jz_k*dM_dz_z;

			if (Ms_k > 0.0) {
				// b_j
				const real b_j = P_k * MU_BOHR / (ELECTRON_CHARGE*Ms_k*(1.0+xi_k*xi_k));
				const real b_j_prime = b_j / (1.0 + alpha_k*alpha_k);
				
				// motion and distortion factors
				real motion_factor, distortion_factor;
				if (do_precess) {
					motion_factor     = -b_j_prime/(Ms_k*Ms_k)*(1.0+alpha_k*xi_k);
					distortion_factor = -b_j_prime/(Ms_k)*(xi_k-alpha_k);
				} else {
					motion_factor     = -b_j_prime/(Ms_k*Ms_k)*(alpha_k*xi_k);
					distortion_factor = -b_j_prime/(Ms_k)*(-alpha_k);
				}

				// c = M x j_grad_M
				real cx, cy, cz;
				gpu_cross<real>(&cx, &cy, &cz, Mx_k, My_k, Mz_k, j_grad_M_x, j_grad_M_y, j_grad_M_z);

				// d = M x c = M x (M x j_grad_M)
				real dx, dy, dz;
				gpu_cross<real>(&dx, &dy, &dz, Mx_k, My_k, Mz_k, cx, cy, cz);

				// add to llg terms
				dmdt_x[k] = motion_factor*dx + distortion_factor*cx;
				dmdt_y[k] = motion_factor*dy + distortion_factor*cy;
				dmdt_z[k] = motion_factor*dz + distortion_factor*cz;
			} else {
				dmdt_x[k] = 0.0;
				dmdt_y[k] = 0.0;
				dmdt_z[k] = 0.0;
			}
		}
	}
}

/////////////////////////////////////////////// Spin torque kernel wrappers //////////////////////////////////////////////////

template <typename real, bool do_precess>
static void fdm_zhangli_cuda_2d(
	int dim_x, int dim_y, 
	double delta_x, double delta_y,
	const Matrix &P, const Matrix &xi, const VectorMatrix &j,
	const Matrix &Ms,
	const Matrix &alpha,
	const VectorMatrix &M,
	VectorMatrix &dM)
{
	typename VectorMatrix_const_cuda_accessor<real>::t M_acc(M), j_acc(j);
	typename VectorMatrix_cuda_accessor<real>::t dM_acc(dM);
	typename Matrix_const_cuda_accessor<real>::t Ms_acc(Ms);
	typename Matrix_const_cuda_accessor<real>::t alpha_acc(alpha), P_acc(P), xi_acc(xi);

	const dim3 grid_dim((dim_x+BLOCK_SIZE_X-1) / BLOCK_SIZE_X, (dim_y+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y);
	const dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	kernel_zhangli_2d_naive<real, do_precess><<<grid_dim, block_dim>>>(
		M_acc.ptr_x(), M_acc.ptr_y(), M_acc.ptr_z(),
		Ms_acc.ptr(),
		alpha_acc.ptr(),
		dM_acc.ptr_x(), dM_acc.ptr_y(), dM_acc.ptr_z(),
		P_acc.ptr(), xi_acc.ptr(), j_acc.ptr_x(), j_acc.ptr_y(), j_acc.ptr_z(),
		dim_x, dim_y, delta_x, delta_y
	);
	checkCudaLastError("fdm_zhangli_cuda_2d(): kernel execution failed!");
	CUDA_THREAD_SYNCHRONIZE();
}

template <typename real, bool do_precess>
static void fdm_zhangli_cuda_3d(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	const Matrix &P, const Matrix &xi, const VectorMatrix &j,
	const Matrix &Ms,
	const Matrix &alpha,
	const VectorMatrix &M,
	VectorMatrix &dM)
{
	typename VectorMatrix_const_cuda_accessor<real>::t M_acc(M), j_acc(j);
	typename VectorMatrix_cuda_accessor<real>::t dM_acc(dM);
	typename Matrix_const_cuda_accessor<real>::t Ms_acc(Ms);
	typename Matrix_const_cuda_accessor<real>::t alpha_acc(alpha), P_acc(P), xi_acc(xi);

	const dim3 grid_dim((dim_x+BLOCK_SIZE_X-1) / BLOCK_SIZE_X, (dim_y+BLOCK_SIZE_Y-1) / BLOCK_SIZE_Y);
	const dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
	kernel_zhangli_3d_naive<real, do_precess><<<grid_dim, block_dim>>>(
		M_acc.ptr_x(), M_acc.ptr_y(), M_acc.ptr_z(),
		Ms_acc.ptr(),
		alpha_acc.ptr(),
		dM_acc.ptr_x(), dM_acc.ptr_y(), dM_acc.ptr_z(),
		P_acc.ptr(), xi_acc.ptr(), j_acc.ptr_x(), j_acc.ptr_y(), j_acc.ptr_z(),
		dim_x, dim_y, dim_z, delta_x, delta_y, delta_z
	);
	checkCudaLastError("fdm_zhangli_cuda_3d(): kernel execution failed!");
	CUDA_THREAD_SYNCHRONIZE();
}

///////////////////////////////////////// Zhang & Li main GPU interface ////////////////////////////////////////

void fdm_zhangli_cuda(
	int dim_x, int dim_y, int dim_z,
	double delta_x, double delta_y, double delta_z,
	bool do_precess,
	const Matrix &P, const Matrix &xi,
	const Matrix &Ms, const Matrix &alpha,
        const VectorMatrix &j, const VectorMatrix &M, 
	VectorMatrix &dM,
	bool cuda64)
{
	const bool is_3d = (dim_z > 1);

#ifdef HAVE_CUDA_64
	if (cuda64)
	if (is_3d) {
		if (do_precess) {
			fdm_zhangli_cuda_3d<double,  true>(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, P, xi, j, Ms, alpha, M, dM);
		} else {
			fdm_zhangli_cuda_3d<double, false>(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, P, xi, j, Ms, alpha, M, dM);
		}
	} else {
		if (do_precess) {
			fdm_zhangli_cuda_2d<double,  true>(dim_x, dim_y, delta_x, delta_y, P, xi, j, Ms, alpha, M, dM);
		} else {
			fdm_zhangli_cuda_2d<double, false>(dim_x, dim_y, delta_x, delta_y, P, xi, j, Ms, alpha, M, dM);
		}
	}
	else
#endif
	if (is_3d) {
		if (do_precess) {
			fdm_zhangli_cuda_3d<float, true>(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, P, xi, j, Ms, alpha, M, dM);
		} else {
			fdm_zhangli_cuda_3d<float, false>(dim_x, dim_y, dim_z, delta_x, delta_y, delta_z, P, xi, j, Ms, alpha, M, dM);
		}
	} else {
		if (do_precess) {
			fdm_zhangli_cuda_2d<float, true>(dim_x, dim_y, delta_x, delta_y, P, xi, j, Ms, alpha, M, dM);
		} else {
			fdm_zhangli_cuda_2d<float, false>(dim_x, dim_y, delta_x, delta_y, P, xi, j, Ms, alpha, M, dM);
		}
	}
}
