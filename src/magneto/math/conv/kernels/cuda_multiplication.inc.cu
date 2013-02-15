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

// Complex multiplication: res = a*b + c*d + e*f
static __inline__ __device__ void gpu_mul3(
	float *res_r, float *res_i,
	float ar, float ai, float br, float bi, float cr, float ci,
	float dr, float di, float er, float ei, float fr, float fi)
{
	(*res_r) = ar*br - ai*bi + cr*dr - ci*di + er*fr - ei*fi;
	(*res_i) = ai*br + ar*bi + ci*dr + cr*di + ei*fr + er*fi;
}

// Complex multiplication: res = a*b + c*d
static __inline__ __device__ void gpu_mul2(
	float *res_r, float *res_i,
	float ar, float ai,
	float br, float bi,
	float cr, float ci,
	float dr, float di)
{
	(*res_r) = ar*br - ai*bi + cr*dr - ci*di;
	(*res_i) = ai*br + ar*bi + ci*dr + cr*di;
}

// H = N*M where N is (3x3) symmetric matrix, i.e. N_ij = N_ji
static __inline__ __device__ void symmetric_tensor_multiplication(
	float Nxx_re, float Nxx_im,
	float Nxy_re, float Nxy_im,
	float Nxz_re, float Nxz_im,
	float Nyy_re, float Nyy_im,
	float Nyz_re, float Nyz_im,
	float Nzz_re, float Nzz_im,

	float Mx_re, float Mx_im, 
	float My_re, float My_im, 
	float Mz_re, float Mz_im,

	float *Hx_re, float *Hx_im, 
	float *Hy_re, float *Hy_im, 
	float *Hz_re, float *Hz_im
) {
	gpu_mul3(Hx_re, Hx_im,              // Hx = 
	     Mx_re, Mx_im, Nxx_re, Nxx_im,  //     Nxx*Mx
	     My_re, My_im, Nxy_re, Nxy_im,  //   + Nxy*My
	     Mz_re, Mz_im, Nxz_re, Nxz_im); //   + Nxz*Mz
		     
	gpu_mul3(Hy_re, Hy_im,              // Hy = 
	     Mx_re, Mx_im, Nxy_re, Nxy_im,  //     Nxy*Mx
	     My_re, My_im, Nyy_re, Nyy_im,  //   + Nyy*My
	     Mz_re, Mz_im, Nyz_re, Nyz_im); //   + Nyz*Mz

	gpu_mul3(Hz_re, Hz_im,              // Hz = 
	     Mx_re, Mx_im, Nxz_re, Nxz_im,  //     Nxz*Mx
	     My_re, My_im, Nyz_re, Nyz_im,  //   + Nyz*My
	     Mz_re, Mz_im, Nzz_re, Nzz_im); //   + Nzz*Mz
}

// H = N*M where N is (3x3) antisymmetric matrix, i.e. N_ij = -N_ji (=> N_ii=0)
static __inline__ __device__ void antisymmetric_tensor_multiplication(
	float Nxy_re, float Nxy_im,
	float Nxz_re, float Nxz_im,
	float Nyz_re, float Nyz_im,

	float Mx_re, float Mx_im, 
	float My_re, float My_im, 
	float Mz_re, float Mz_im,

	float *Hx_re, float *Hx_im, 
	float *Hy_re, float *Hy_im, 
	float *Hz_re, float *Hz_im
) {
	gpu_mul2(Hx_re, Hx_im,                    // Hx = 
	         My_re, My_im, +Nxy_re, +Nxy_im,  //      My*Nxy
	         Mz_re, Mz_im, +Nxz_re, +Nxz_im); //      Mz*Nxz

	gpu_mul2(Hy_re, Hy_im,                    // Hy = 
	         Mx_re, Mx_im, -Nxy_re, -Nxy_im,  //      Mx*Nyx = Mx*-Nxy
	         Mz_re, Mz_im, +Nyz_re, +Nyz_im); //      Mz*Nzz
	
	gpu_mul2(Hz_re, Hz_im,                    // Hz = 
	         Mx_re, Mx_im, -Nxz_re, -Nxz_im,  //      Mx*Nzx = Mx*-Nxz
	         My_re, My_im, -Nyz_re, -Nyz_im); //      My*Nzy = My*-Nyz
}
