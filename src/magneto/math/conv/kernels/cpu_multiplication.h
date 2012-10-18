#ifndef CPU_MULTIPLICATION_H
#define CPU_MULTIPLICATION_H

void cpu_multiplication_symmetric(
	int num_elements,
	const double *Nxxr, const double *Nxyr, const double *Nxzr, const double *Nyyr, const double *Nyzr, const double *Nzzr, /*in*/
	const double *Nxxi, const double *Nxyi, const double *Nxzi, const double *Nyyi, const double *Nyzi, const double *Nzzi, /*in*/
	double *Mx, double *My, double *Mz); /*inout*/

void cpu_multiplication_antisymmetric(
	int num_elements,
	const double *Nxyr, const double *Nxzr, const double *Nyzr, /*in*/
	const double *Nxyi, const double *Nxzi, const double *Nyzi, /*in*/
	double *Mx, double *My, double *Mz); /*inout*/

#endif
