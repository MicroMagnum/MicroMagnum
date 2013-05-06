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
