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
