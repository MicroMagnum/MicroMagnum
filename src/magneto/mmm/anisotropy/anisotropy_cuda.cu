#include "config.h"
#include "anisotropy_cuda.h"
#include "mmm/constants.h"

#include <stdexcept>

static const int GRID_SIZE = 32; // TODO: Setze auf Anzahl Cores pro Multiprozessor
static const int BLOCK_SIZE = 128;

// hack: use sum routines from matrix library..
template <typename real> 
double cuda_compute_sum(const real *src, int N)
{
	extern double cuda_sum(const real *src, int N);
	return cuda_sum(src, N);
}

template <typename real>
__global__ void kernel_uniaxial_anisotropy(
	const real *Mx_ptr, const real *My_ptr, const real *Mz_ptr,
	const real *ax_ptr, const real *ay_ptr, const real *az_ptr,
	const real *Ms_ptr, const real *k_ptr,
	real *Hx_ptr, real *Hy_ptr, real *Hz_ptr, real *E_ptr,
	int N)
{
	const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		// Anisotropy field at cell i
		const real Ms = Ms_ptr[i], k = k_ptr[i];
		if (Ms == 0.0 || k == 0.0) {
			Hx_ptr[i] = 0.0;
			Hy_ptr[i] = 0.0;
			Hz_ptr[i] = 0.0;
			 E_ptr[i] = 0.0;
		} else {
			// compute d = cos(angle between M and unit axis)
			const real Mx = Mx_ptr[i], My = My_ptr[i], Mz = Mz_ptr[i];
			const real ax = ax_ptr[i], ay = ay_ptr[i], az = az_ptr[i];
			const real  d = (Mx*ax + My*ay + Mz*az) / Ms;
			
			const real f = 2.0 * k * d / real(MU0) / Ms;
			Hx_ptr[i] = f * ax;
			Hy_ptr[i] = f * ay;
			Hz_ptr[i] = f * az;
			 E_ptr[i] = k * (1.0 - d*d);
		}
	}
}

template <typename real>
__global__ void kernel_cubic_anisotropy(
	const real *Mx_ptr, const real *My_ptr, const real *Mz_ptr,
	const real *ax1_ptr, const real *ay1_ptr, const real *az1_ptr,
	const real *ax2_ptr, const real *ay2_ptr, const real *az2_ptr,
	const real *Ms_ptr, const real *k_ptr,
	real *Hx_ptr, real *Hy_ptr, real *Hz_ptr, real *E_ptr,
	int N)
{
	const int     tid = blockDim.x * blockIdx.x + threadIdx.x;
	const int threadN = blockDim.x * gridDim.x;

	for (int i=tid; i<N; i+=threadN) {
		// Parameters for cell i
		const real Mx = Mx_ptr[i], My = My_ptr[i], Mz = Mz_ptr[i];
		const real ax1 = ax1_ptr[i], ay1 = ay1_ptr[i], az1 = az1_ptr[i];
		const real ax2 = ax2_ptr[i], ay2 = ay2_ptr[i], az2 = az2_ptr[i];
		const real Ms = Ms_ptr[i], k = k_ptr[i];

		// Third axis: axis3 = cross(axis1, axis2)
		const real ax3 = ay1*az2 - az1*ay2;
		const real ay3 = az1*ax2 - ax1*az2;
		const real az3 = ax1*ay2 - ay1*ax2;

		if (Ms == 0.0f) {
			Hx_ptr[i] = 0.0;
			Hy_ptr[i] = 0.0;
			Hz_ptr[i] = 0.0;
	 		 E_ptr[i] = 0.0;
		} else {
			// a1, a2, a3: coordinates of unit magnetization in coordinate system with base (axis1, axis2, axis3).
			const real a1 = (ax1*Mx + ay1*My + az1*Mz) / Ms;
			const real a2 = (ax2*Mx + ay2*My + az2*Mz) / Ms;
			const real a3 = (ax3*Mx + ay3*My + az3*Mz) / Ms;

			const real f = -2.0 * k / real(MU0) / Ms;
			const real Hx = f * ((a2*a2+a3*a3)*a1*ax1 + (a1*a1+a3*a3)*a2*ax2 + (a1*a1+a2*a2)*a3*ax3);
			const real Hy = f * ((a2*a2+a3*a3)*a1*ay1 + (a1*a1+a3*a3)*a2*ay2 + (a1*a1+a2*a2)*a3*ay3);
			const real Hz = f * ((a2*a2+a3*a3)*a1*az1 + (a1*a1+a3*a3)*a2*az2 + (a1*a1+a2*a2)*a3*az3);

			Hx_ptr[i] = Hx;
			Hy_ptr[i] = Hy;
			Hz_ptr[i] = Hz;
			 E_ptr[i] = k * (a1sq*a2sq+a1sq*a3sq+a2sq*a3sq);
		}
	}
}

//////////////////////////////////////////////////////////////////////////////
// MAIN ROUTINE THAT CALLS THE KERNELS                                      //
//////////////////////////////////////////////////////////////////////////////

template <typename real>
double uniaxial_anisotropy_cuda_impl(
	const VectorMatrix &axis,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	typename VectorMatrix_cuda_accessor<real>::t H_acc(H);
	typename VectorMatrix_const_cuda_accessor<real>::t M_acc(M), axis_acc(axis);
	typename Matrix_const_cuda_accessor<real>::t Ms_acc(Ms), k_acc(k);

	const real *ax_ptr = axis_acc.ptr_x();
	const real *ay_ptr = axis_acc.ptr_y();
	const real *az_ptr = axis_acc.ptr_z();
	const real *k_ptr  = k_acc.ptr();
	const real *Ms_ptr = Ms_acc.ptr();
	const real *Mx_ptr = M_acc.ptr_x();
	const real *My_ptr = M_acc.ptr_y();
	const real *Mz_ptr = M_acc.ptr_z();
	      real *Hx_ptr = H_acc.ptr_x();
	      real *Hy_ptr = H_acc.ptr_y();
	      real *Hz_ptr = H_acc.ptr_z();

	real *E_ptr;
	const int N = M.size();
	const cudaError_t err = cudaMalloc((void**)&E_ptr, N * sizeof(real));
	if (err != cudaSuccess) throw std::runtime_error("cubic_anisotropy_cuda_impl: could not allocate memory on GPU");

	kernel_uniaxial_anisotropy<<<GRID_SIZE, BLOCK_SIZE>>>(
		Mx_ptr, My_ptr, Mz_ptr, 
		ax_ptr, ay_ptr, az_ptr, Ms_ptr, k_ptr,
		Hx_ptr, Hy_ptr, Hz_ptr, E_ptr,
		N
	);
	checkCudaLastError("kernel_uniaxial_anisotropy() execution failed");
	CUDA_THREAD_SYNCHRONIZE();

	const double energy_sum = cuda_compute_sum<real>(E_ptr, N);
	cudaFree(E_ptr);
	return energy_sum;
}

template <typename real>
double cubic_anisotropy_cuda_impl(
	const VectorMatrix &axis1,
	const VectorMatrix &axis2,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H)
{
	typename VectorMatrix_cuda_accessor<real>::t H_acc(H);
	typename VectorMatrix_const_cuda_accessor<real>::t M_acc(M), axis1_acc(axis1), axis2_acc(axis2);
	typename Matrix_const_cuda_accessor<real>::t Ms_acc(Ms), k_acc(k);

	const real *ax1_ptr = axis1_acc.ptr_x();
	const real *ay1_ptr = axis1_acc.ptr_y();
	const real *az1_ptr = axis1_acc.ptr_z();
	const real *ax2_ptr = axis2_acc.ptr_x();
	const real *ay2_ptr = axis2_acc.ptr_y();
	const real *az2_ptr = axis2_acc.ptr_z();
	const real *k_ptr  = k_acc.ptr();
	const real *Ms_ptr = Ms_acc.ptr();
	const real *Mx_ptr = M_acc.ptr_x();
	const real *My_ptr = M_acc.ptr_y();
	const real *Mz_ptr = M_acc.ptr_z();
	      real *Hx_ptr = H_acc.ptr_x();
	      real *Hy_ptr = H_acc.ptr_y();
	      real *Hz_ptr = H_acc.ptr_z();

	real *E_ptr = 0;
	const int N = M.size();
	const cudaError_t err = cudaMalloc((void**)&E_ptr, N * sizeof(real));
	if (err != cudaSuccess) throw std::runtime_error("cubic_anisotropy_cuda_impl: could not allocate memory on GPU");

	kernel_cubic_anisotropy<<<GRID_SIZE, BLOCK_SIZE>>>(
		Mx_ptr, My_ptr, Mz_ptr, 
		ax1_ptr, ay1_ptr, az1_ptr, ax2_ptr, ay2_ptr, az2_ptr, Ms_ptr, k_ptr,
		Hx_ptr, Hy_ptr, Hz_ptr, E_ptr,
		N
	);
	checkCudaLastError("kernel_cubic_anisotropy() execution failed");
	CUDA_THREAD_SYNCHRONIZE();

	const double energy_sum = cuda_compute_sum<real>(E_ptr, N);
	cudaFree(E_ptr);
	return energy_sum;
}

double uniaxial_anisotropy_cuda(
	const VectorMatrix &axis,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H,
	bool cuda64)
{
#ifdef HAVE_CUDA_64
	if (cuda64) 
	return uniaxial_anisotropy_cuda_impl<double>(axis, k, Ms, M, H); 
	else
#endif
	return uniaxial_anisotropy_cuda_impl<float>(axis, k, Ms, M, H);
}

double cubic_anisotropy_cuda(
	const VectorMatrix &axis1,
	const VectorMatrix &axis2,
	const       Matrix &k,
	const       Matrix &Ms,
	const VectorMatrix &M,
	VectorMatrix &H,
	bool cuda64)
{
#ifdef HAVE_CUDA_64
	if (cuda64)
	return cubic_anisotropy_cuda_impl<double>(axis1, axis2, k, Ms, M, H);
	else
#endif
	return cubic_anisotropy_cuda_impl<float>(axis1, axis2, k, Ms, M, H);
}
