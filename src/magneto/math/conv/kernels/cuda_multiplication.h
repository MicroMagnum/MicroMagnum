#ifndef CUDA_MULTIPLICATION_H
#define CUDA_MULTIPLICATION_H

void cuda_multiplication_symmetric(
	int num_elements,
	const float *Nxxr, const float *Nxyr, const float *Nxzr, const float *Nyyr, const float *Nyzr, const float *Nzzr, /*in*/
	const float *Nxxi, const float *Nxyi, const float *Nxzi, const float *Nyyi, const float *Nyzi, const float *Nzzi, /*in*/
	float *Mx, float *My, float *Mz); /*inout*/

void cuda_multiplication_antisymmetric(
	int num_elements,
	const float *Nxyr, const float *Nxzr, const float *Nyzr, /*in*/
	const float *Nxyi, const float *Nxzi, const float *Nyzi, /*in*/
	float *Mx, float *My, float *Mz); /*inout*/

void cuda_multiplication_scalar_product(
	int num_elements,
	const float *Sxr, const float *Syr, const float *Szr, /*in*/
	const float *Sxi, const float *Syi, const float *Szi, /*in*/
	float *Mx /*inout*/, const float *My /*out*/, const float *Mz /*out*/);

#endif
