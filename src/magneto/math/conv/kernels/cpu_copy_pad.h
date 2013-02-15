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

#ifndef CPU_COPY_PAD_H
#define CPU_COPY_PAD_H

/*
 *  Input size: dim_x * dim_y * dim_z
 * Output size: exp_x * exp_y * exp_z
 */
void cpu_copy_pad_r2c(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double * in_x, const double * in_y, const double * in_z, 
	      double *out_x,       double *out_y,       double *out_z
);

/*
 *  Input size: dim_x * dim_y * dim_z
 * Output size: exp_x * exp_y * exp_z
 */
void cpu_copy_pad_r2r(
	int dim_x, int dim_y, int dim_z,
	int exp_x,
	const double * in_x, const double * in_y, const double * in_z,
	      double *out_x,       double *out_y,       double *out_z
);

#endif
