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

/*
 * This file was developed by Benjamin Krueger.
 */

#ifndef TENSOR_INTEGRALS_H
#define TENSOR_INTEGRALS_H

#include "config.h"
#include <iostream>
#include <limits>
#include <cassert>
#include <cmath>
#include <math.h>

#include "mmm/constants.h"

namespace tensor_integrals
{
	static double expansionradius = 30.0;
	static double expansionradius_oersted = 30.0;
	static double expansionradius_diag = 30.0;
	static double expansionradius_nondiag = 30.0;
//	static double max_aspect_ratio = 100;
	const int SERIES_POSITION_CENTER = 0;
	const int SERIES_POSITION_BORDER = 1;

	template <class T>
	inline T Power(T a, int b)
	{
		if (b < 0) return 1.0 / Power(a, -b);

		T result = 1.0;
		for (int i=0; i<b; ++i) result *= a;
		return result;
	}

	template <class T>
	inline T ArcTan(T a)
	{
		return atan(a);
	}

	// Note: my_log1p and my_atanh copied from the GNU scientific library (sys/invhyp.c, sys/log1p.c) which is licensed under the GPL.
	//       (Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007 Brian Gough)

	template <class T>
	T gsl_log1p(const T x)
	{
		volatile T y, z;
		y = 1 + x;
		z = y - 1;
		return std::log(y) - (z-x)/y ;  /* cancels errors with IEEE arithmetic */
	}

	template <class T>
	T gsl_atanh(const T x)
	{
		T a = std::abs(x);
		T s = (x < 0) ? -1 : 1;

		if (a > 1) {
			return std::numeric_limits<T>::signaling_NaN();
		} else if (a == 1) {
			return (x < 0) ? -std::numeric_limits<T>::infinity() : +std::numeric_limits<T>::infinity();
		} else if (a >= 0.5) {
			return s * 0.5 * gsl_log1p(2 * a / (1 - a));
		} else if (a > std::numeric_limits<T>::epsilon()) {
			return s * 0.5 * gsl_log1p(2 * a + 2 * a * a / (1 - a));
		} else {
			return x;
		}
	}

	template <class T>
	inline T ArcTanh(T a)
	{
#ifdef _MSC_VER
		return gsl_atanh<T>(a); // For Visual C++: Use the above implementation of atanh
#else
		return atanh(a); // from math.h (atanh is NOT in pre-2011 C++ standard, but can be used from GCC anyway)
#endif
	}

	inline double Factorial(int x)
	{
		static const double f[13] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600};
		if (x < 13) return f[x];

		double y = f[12];
		for (int i = 13; i <= x; i++) y *= i;
		return y;		
	}

	template <class T>
	inline T Abs(T x)
	{
		return fabs(x);		
	}

	template <class T>
	inline T Sqrt(T x)
	{
		return sqrt(x);		
	}

	template <class T>
	inline T f(int a, int b, int c, T lx, T ly, T lz, int i, int j, int k)
	{
		if (a > b) return f(b, a, c, ly, lx, lz, j, i, k);
		if (b > c) return f(a, c, b, lx, lz, ly, i, k, j);

		const T x = i*lx;
		const T y = j*ly;
		const T z = k*lz;
		const T R = std::sqrt(x*x+y*y+z*z);

		if (a == 0 && b == 0 && c == 1) {
			if (k == 0) return 0;
			if (i == 0 && j == 0) return (-3*R*(Power(x,2) + Power(y,2))*z + 2*R*Power(z,3) - 4*x*y*(3*Power(z,2)*ArcTan((x*y)/(R*z))) - 4*x*z*(-3*Power(y,2) + Power(z,2))*ArcTanh(x/R) - 4*y*z*(-3*Power(x,2) + Power(z,2))*ArcTanh(y/R))/24.;
			if (i == 0) return (-3*R*(Power(x,2) + Power(y,2))*z + 2*R*Power(z,3) - 4*x*y*(3*Power(z,2)*ArcTan((x*y)/(R*z)) + Power(y,2)*ArcTan((x*z)/(R*y))) - 4*x*z*(-3*Power(y,2) + Power(z,2))*ArcTanh(x/R) - 4*y*z*(-3*Power(x,2) + Power(z,2))*ArcTanh(y/R) - (Power(x,4) - 6*Power(x,2)*Power(y,2) + Power(y,4))*ArcTanh(z/R))/24.;
			if (j == 0) return (-3*R*(Power(x,2) + Power(y,2))*z + 2*R*Power(z,3) - 4*x*y*(3*Power(z,2)*ArcTan((x*y)/(R*z)) + Power(x,2)*ArcTan((y*z)/(R*x))) - 4*x*z*(-3*Power(y,2) + Power(z,2))*ArcTanh(x/R) - 4*y*z*(-3*Power(x,2) + Power(z,2))*ArcTanh(y/R) - (Power(x,4) - 6*Power(x,2)*Power(y,2) + Power(y,4))*ArcTanh(z/R))/24.;
			return (-3*R*(Power(x,2) + Power(y,2))*z + 2*R*Power(z,3) - 4*x*y*(3*Power(z,2)*ArcTan((x*y)/(R*z)) + Power(y,2)*ArcTan((x*z)/(R*y)) + Power(x,2)*ArcTan((y*z)/(R*x))) - 4*x*z*(-3*Power(y,2) + Power(z,2))*ArcTanh(x/R) - 4*y*z*(-3*Power(x,2) + Power(z,2))*ArcTanh(y/R) - (Power(x,4) - 6*Power(x,2)*Power(y,2) + Power(y,4))*ArcTanh(z/R))/24.;
		} else if (a == 0 && b == 0 && c == 2) {
			if (i == 0 && j == 0 && k == 0) return 0;
			if (i == 0 && k == 0) return (-(R*(Power(x,2) + Power(y,2) - 2*Power(z,2))) + 3*x*(y - z)*(y + z)*ArcTanh(x/R))/6.;
			if (j == 0 && k == 0) return (-(R*(Power(x,2) + Power(y,2) - 2*Power(z,2))) + 3*y*(x - z)*(x + z)*ArcTanh(y/R))/6.;
			if (k == 0) return (-(R*(Power(x,2) + Power(y,2) - 2*Power(z,2))) + 3*x*(y - z)*(y + z)*ArcTanh(x/R) + 3*y*(x - z)*(x + z)*ArcTanh(y/R))/6.;
			return (-(R*(Power(x,2) + Power(y,2) - 2*Power(z,2))) - 6*x*y*z*ArcTan((x*y)/(R*z)) + 3*x*(y - z)*(y + z)*ArcTanh(x/R) + 3*y*(x - z)*(x + z)*ArcTanh(y/R))/6.;
		} else if (a == 0 && b == 0 && c == 3) {
			return R*z - x*(y*ArcTan((x*y)/(R*z)) + z*ArcTanh(x/R)) - y*z*ArcTanh(y/R);
		} else if (a == 0 && b == 0 && c == 4) {
			return 2*R - x*ArcTanh(x/R) - y*ArcTanh(y/R);
		} else if (a == 0 && b == 0 && c == 5) {
			return R*z*(1/(Power(x,2) + Power(z,2)) + 1/(Power(y,2) + Power(z,2)));
		} else if (a == 0 && b == 0 && c == 6) {
			return (Power(x,2)*Power(y,2)*Power(Power(x,2) + Power(y,2),2) - (Power(x,6) - 5*Power(x,4)*Power(y,2) - 5*Power(x,2)*Power(y,4) + Power(y,6))*Power(z,2) - (Power(x,4) - 6*Power(x,2)*Power(y,2) + Power(y,4))*Power(z,4))/ (R*Power(Power(x,2) + Power(z,2),2)*Power(Power(y,2) + Power(z,2),2));
		} else if (a == 0 && b == 0 && c == 7) {
			return (z*((-8*Power(x,2)*Power(y,4))/Power(Power(x,2) + Power(z,2),3) + (2*Power(y,2)*(-6*Power(x,2) + Power(y,2)))/Power(Power(x,2) + Power(z,2),2) - (3*(x - y)*(x + y))/(Power(x,2) + Power(z,2)) - (8*Power(x,4)*Power(y,2))/Power(Power(y,2) + Power(z,2),3) + (2*Power(x,2)*(Power(x,2) - 6*Power(y,2)))/Power(Power(y,2) + Power(z,2),2) + (3*(x - y)*(x + y))/(Power(y,2) + Power(z,2))))/Power(R,3);
		} else if (a == 0 && b == 0 && c == 8) {
			return (3*((-16*Power(x,4)*Power(y,6))/Power(Power(x,2) + Power(z,2),4) - (8*(5*Power(x,4)*Power(y,4) - 2*Power(x,2)*Power(y,6)))/Power(Power(x,2) + Power(z,2),3) - (2*(15*Power(x,4)*Power(y,2) - 20*Power(x,2)*Power(y,4) + Power(y,6)))/ Power(Power(x,2) + Power(z,2),2) - (5*(Power(x,4) - 6*Power(x,2)*Power(y,2) + Power(y,4)))/ (Power(x,2) + Power(z,2)) - (16*Power(x,6)*Power(y,4))/Power(Power(y,2) + Power(z,2),4) + (8*(2*Power(x,6)*Power(y,2) - 5*Power(x,4)*Power(y,4)))/Power(Power(y,2) + Power(z,2),3) - (2*(Power(x,6) - 20*Power(x,4)*Power(y,2) + 15*Power(x,2)*Power(y,4)))/ Power(Power(y,2) + Power(z,2),2) - (5*(Power(x,4) - 6*Power(x,2)*Power(y,2) + Power(y,4)))/ (Power(y,2) + Power(z,2))))/Power(R,5);
		} else if (a == 0 && b == 0 && c == 9) {
			return (3*z*(8*Power(x,18)*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4)) + 4*Power(x,16)*(7*Power(y,2) + 17*Power(z,2))*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4)) + 5*Power(x,14)*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4))*(7*Power(y,4) + 42*Power(y,2)*Power(z,2) + 51*Power(z,4)) + Power(y,4)*Power(z,4)*Power(Power(y,2) + Power(z,2),4)*(8*Power(y,6) + 36*Power(y,4)*Power(z,2) + 63*Power(y,2)*Power(z,4) + 70*Power(z,6)) + 5*Power(x,12)*(21*Power(y,10) + 210*Power(y,8)*Power(z,2) + 266*Power(y,6)*Power(z,4) - 732*Power(y,4)*Power(z,6) - 951*Power(y,2)*Power(z,8) + 114*Power(z,10)) - Power(x,2)*Power(y,2)*Power(z,2)*Power(Power(y,2) + Power(z,2),3)*(80*Power(y,10) + 412*Power(y,8)*Power(z,2) + 864*Power(y,6)*Power(z,4) + 847*Power(y,4)*Power(z,6) + 455*Power(y,2)*Power(z,8) + 420*Power(z,10)) + 5*Power(x,10)*(21*Power(y,12) + 196*Power(y,10)*Power(z,2) + 672*Power(y,8)*Power(z,4) + 308*Power(y,6)*Power(z,6) - 1423*Power(y,4)*Power(z,8) - 1200*Power(y,2)*Power(z,10) + 162*Power(z,12)) + 5*Power(x,4)*Power(Power(y,2) + Power(z,2),2)*(8*Power(y,14) - 4*Power(y,12)*Power(z,2) - 158*Power(y,10)*Power(z,4) - 412*Power(y,8)*Power(z,6) - 441*Power(y,6)*Power(z,8) - 448*Power(y,4)*Power(z,10) - 371*Power(y,2)*Power(z,12) + 14*Power(z,14)) + 7*Power(x,6)*(Power(y,2) + Power(z,2))*(20*Power(y,14) + 80*Power(y,12)*Power(z,2) + 110*Power(y,10)*Power(z,4) + 110*Power(y,8)*Power(z,6) - 85*Power(y,6)*Power(z,8) - 675*Power(y,4)*Power(z,10) - 545*Power(y,2)*Power(z,12) + 49*Power(z,14)) + Power(x,8)*(175*Power(y,14) + 1050*Power(y,12)*Power(z,2) + 3360*Power(y,10)*Power(z,4) + 5600*Power(y,8)*Power(z,6) + 175*Power(y,6)*Power(z,8) - 8710*Power(y,4)*Power(z,10) - 5190*Power(y,2)*Power(z,12) + 708*Power(z,14))))/(Power(R,7)*Power(Power(x,2) + Power(z,2),5)*Power(Power(y,2) + Power(z,2),5));
		} else if (a == 0 && b == 1 && c == 1) {
			if (j == 0 || k == 0) return 0;
			if (i == 0) return (-2*y*z*R - Power(z,3)*ArcTanh(y/R) - Power(y,3)*ArcTanh(z/R))/6.;
			return (-2*R*y*z - 3*x*Power(z,2)*ArcTan((x*y)/(R*z)) - 3*x*Power(y,2)*ArcTan((x*z)/(R*y)) - Power(x,3)*ArcTan((y*z)/(R*x)) + 6*x*y*z*ArcTanh(x/R) + (3*Power(x,2)*z - Power(z,3))*ArcTanh(y/R) - y*(-3*Power(x,2) + Power(y,2))*ArcTanh(z/R))/6.;
		} else if (a == 0 && b == 1 && c == 2) {
			return (-(R*y) - 2*x*z*ArcTan((x*y)/(R*z)) + 2*x*y*ArcTanh(x/R) + (x - z)*(x + z)*ArcTanh(y/R))/2.;
		} else if (a == 0 && b == 1 && c == 3) {
			return -(x*ArcTan((x*y)/(R*z))) - z*ArcTanh(y/R);
		} else if (a == 0 && b == 1 && c == 4) {
			return (R*y)/(Power(y,2) + Power(z,2)) - ArcTanh(y/R);
		} else if (a == 0 && b == 1 && c == 5) {
			return (y*z*(1/(Power(x,2) + Power(z,2)) - (2*Power(x,2) + Power(y,2) + Power(z,2))/Power(Power(y,2) + Power(z,2),2)))/ R;
		} else if (a == 0 && b == 1 && c == 6) {
			return (y*((2*Power(x,2)*Power(y,2))/Power(Power(x,2) + Power(z,2),2) + (3*Power(x,2) - Power(y,2))/(Power(x,2) + Power(z,2)) - (8*Power(x,4)*Power(y,2))/Power(Power(y,2) + Power(z,2),3) + (6*Power(x,2)*(Power(x,2) - 2*Power(y,2)))/Power(Power(y,2) + Power(z,2),2) + (9*Power(x,2) - 3*Power(y,2))/(Power(y,2) + Power(z,2))))/Power(R,3);
		} else if (a == 0 && b == 1 && c == 7) {
			return (y*z*((-8*Power(x,2)*Power(y,4))/Power(Power(x,2) + Power(z,2),3) + (2*Power(y,2)*(-10*Power(x,2) + Power(y,2)))/Power(Power(x,2) + Power(z,2),2) + (5*(-3*Power(x,2) + Power(y,2)))/(Power(x,2) + Power(z,2)) + (48*Power(x,6)*Power(y,2))/Power(Power(y,2) + Power(z,2),4) - (24*Power(x,4)*(Power(x,2) - 5*Power(y,2)))/Power(Power(y,2) + Power(z,2),3) + (30*Power(x,2)*(-2*Power(x,2) + 3*Power(y,2)))/Power(Power(y,2) + Power(z,2),2) + (15*(-3*Power(x,2) + Power(y,2)))/(Power(y,2) + Power(z,2))))/Power(R,5);
		} else if (a == 0 && b == 2 && c == 1) {
			return (-(R*z) - 2*x*y*ArcTan((x*z)/(R*y)) + 2*x*z*ArcTanh(x/R) + (x - y)*(x + y)*ArcTanh(z/R))/2.;
		} else if (a == 0 && b == 2 && c == 2) {
			return -R + x*ArcTanh(x/R);
		} else if (a == 0 && b == 2 && c == 3) {
			return -((R*z)/(Power(y,2) + Power(z,2)));
		} else if (a == 0 && b == 2 && c == 4) {
			return (-(Power(y,2)*(Power(x,2) + Power(y,2))) + (x - y)*(x + y)*Power(z,2))/(R*Power(Power(y,2) + Power(z,2),2));
		} else if (a == 0 && b == 2 && c == 5) {
			return (z*(Power(x,4)*(6*Power(y,2) - 2*Power(z,2)) + 3*Power(y,2)*Power(Power(y,2) + Power(z,2),2) + Power(x,2)*(9*Power(y,4) + 6*Power(y,2)*Power(z,2) - 3*Power(z,4))))/ (Power(R,3)*Power(Power(y,2) + Power(z,2),3));
		} else if (a == 0 && b == 2 && c == 6) {
			return (3*(Power(y,2)*(Power(y,2) - 4*Power(z,2))*Power(Power(y,2) + Power(z,2),3) + 2*Power(x,6)*(Power(y,4) - 6*Power(y,2)*Power(z,2) + Power(z,4)) + 2*Power(x,2)*Power(Power(y,2) + Power(z,2),2)*(2*Power(y,4) - 11*Power(y,2)*Power(z,2) + 2*Power(z,4)) + 5*Power(x,4)*(Power(y,6) - 5*Power(y,4)*Power(z,2) - 5*Power(y,2)*Power(z,4) + Power(z,6))))/ (Power(R,5)*Power(Power(y,2) + Power(z,2),4));
		} else if (a == 0 && b == 2 && c == 7) {
			return (-3*z*(5*Power(y,2)*(3*Power(y,2) - 4*Power(z,2))*Power(Power(y,2) + Power(z,2),4) + 8*Power(x,8)*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4)) + 28*Power(x,6)*(Power(y,2) + Power(z,2))*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4)) + 10*Power(x,2)*Power(Power(y,2) + Power(z,2),3)*(9*Power(y,4) - 17*Power(y,2)*Power(z,2) + 2*Power(z,4)) + 35*Power(x,4)*(5*Power(y,8) - 14*Power(y,4)*Power(z,4) - 8*Power(y,2)*Power(z,6) + Power(z,8))))/(Power(R,7)*Power(Power(y,2) + Power(z,2),5));
		} else if (a == 0 && b == 2 && c == 8) {
			return (-15*(3*Power(y,2)*Power(Power(y,2) + Power(z,2),5)*(Power(y,4) - 12*Power(y,2)*Power(z,2) + 8*Power(z,4)) + Power(x,4)*Power(Power(y,2) + Power(z,2),3)*(53*Power(y,6) - 786*Power(y,4)*Power(z,2) + 789*Power(y,2)*Power(z,4) - 52*Power(z,6)) + 3*Power(x,2)*Power(Power(y,2) + Power(z,2),4)*(7*Power(y,6) - 99*Power(y,4)*Power(z,2) + 96*Power(y,2)*Power(z,4) - 8*Power(z,6)) + 8*Power(x,10)*(Power(y,6) - 15*Power(y,4)*Power(z,2) + 15*Power(y,2)*Power(z,4) - Power(z,6)) + 63*Power(x,6)*Power(Power(y,2) + Power(z,2),2)*(Power(y,6) - 15*Power(y,4)*Power(z,2) + 15*Power(y,2)*Power(z,4) - Power(z,6)) + 36*Power(x,8)*(Power(y,8) - 14*Power(y,6)*Power(z,2) + 14*Power(y,2)*Power(z,6) - Power(z,8))))/(Power(R,9)*Power(Power(y,2) + Power(z,2),6));
		} else if (a == 0 && b == 3 && c == 1) {
			return -(x*ArcTan((x*z)/(R*y))) - y*ArcTanh(z/R);
		} else if (a == 0 && b == 3 && c == 3) {
			return (y*z*(2*Power(x,2) + Power(y,2) + Power(z,2)))/(R*Power(Power(y,2) + Power(z,2),2));
		} else if (a == 0 && b == 3 && c == 4) {
			return (y*(Power(y,6) - 3*Power(y,2)*Power(z,4) - 2*Power(z,6) + 2*Power(x,4)*(Power(y,2) - 3*Power(z,2)) + 3*Power(x,2)*(Power(y,2) - 3*Power(z,2))*(Power(y,2) + Power(z,2))))/ (Power(R,3)*Power(Power(y,2) + Power(z,2),3));
		} else if (a == 0 && b == 3 && c == 5) {
			return (-3*y*z*(8*Power(x,6)*(y - z)*(y + z) + 15*Power(x,2)*(y - z)*(y + z)*Power(Power(y,2) + Power(z,2),2) + (3*Power(y,2) - 2*Power(z,2))*Power(Power(y,2) + Power(z,2),3) + 20*Power(x,4)*(Power(y,4) - Power(z,4))))/(Power(R,5)*Power(Power(y,2) + Power(z,2),4));
		} else if (a == 0 && b == 4 && c == 4) {
			return -((6*Power(x,6)*(Power(y,4) - 6*Power(y,2)*Power(z,2) + Power(z,4)) + Power(Power(y,2) + Power(z,2),3)*(2*Power(y,4) - 11*Power(y,2)*Power(z,2) + 2*Power(z,4)) + Power(x,2)*Power(Power(y,2) + Power(z,2),2)*(11*Power(y,4) - 68*Power(y,2)*Power(z,2) + 11*Power(z,4)) + 15*Power(x,4)*(Power(y,6) - 5*Power(y,4)*Power(z,2) - 5*Power(y,2)*Power(z,4) + Power(z,6)))/ (Power(R,5)*Power(Power(y,2) + Power(z,2),4)));
		} else if (a == 0 && b == 4 && c == 5) {
			return (3*z*(8*Power(x,8)*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4)) + 28*Power(x,6)*(Power(y,2) + Power(z,2))*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4)) + Power(Power(y,2) + Power(z,2),4)*(12*Power(y,4) - 21*Power(y,2)*Power(z,2) + 2*Power(z,4)) + Power(x,2)*Power(Power(y,2) + Power(z,2),3)*(87*Power(y,4) - 176*Power(y,2)*Power(z,2) + 17*Power(z,4)) + 35*Power(x,4)*(5*Power(y,8) - 14*Power(y,4)*Power(z,4) - 8*Power(y,2)*Power(z,6) + Power(z,8))))/(Power(R,7)*Power(Power(y,2) + Power(z,2),5));
		} else if (a == 0 && b == 4 && c == 6) {
			return (3*(Power(x,4)*Power(Power(y,2) + Power(z,2),3)*(262*Power(y,6) - 3939*Power(y,4)*Power(z,2) + 3936*Power(y,2)*Power(z,4) - 263*Power(z,6)) + 3*Power(x,2)*Power(Power(y,2) + Power(z,2),4)*(33*Power(y,6) - 491*Power(y,4)*Power(z,2) + 494*Power(y,2)*Power(z,4) - 32*Power(z,6)) + Power(Power(y,2) + Power(z,2),5)*(12*Power(y,6) - 159*Power(y,4)*Power(z,2) + 136*Power(y,2)*Power(z,4) - 8*Power(z,6)) + 40*Power(x,10)*(Power(y,6) - 15*Power(y,4)*Power(z,2) + 15*Power(y,2)*Power(z,4) - Power(z,6)) + 315*Power(x,6)*Power(Power(y,2) + Power(z,2),2)*(Power(y,6) - 15*Power(y,4)*Power(z,2) + 15*Power(y,2)*Power(z,4) - Power(z,6)) + 180*Power(x,8)*(Power(y,8) - 14*Power(y,6)*Power(z,2) + 14*Power(y,2)*Power(z,6) - Power(z,8))))/(Power(R,9)*Power(Power(y,2) + Power(z,2),6));
		} else if (a == 1 && b == 1 && c == 2) {
			return -(z*ArcTan((x*y)/(R*z))) + y*ArcTanh(x/R) + x*ArcTanh(y/R);
		} else if (a == 1 && b == 1 && c == 3) {
			return -ArcTan((x*y)/(z*R));
		} else if (a == 1 && b == 1 && c == 4) {
			return (x*y*(1/(Power(x,2) + Power(z,2)) + 1/(Power(y,2) + Power(z,2))))/R;
		} else if (a == 1 && b == 1 && c == 5) {
			return -((x*y*z*(2*Power(x,6) + 3*Power(x,4)*Power(y,2) + 3*Power(x,2)*Power(y,4) + 2*Power(y,6) + (7*Power(x,4) + 12*Power(x,2)*Power(y,2) + 7*Power(y,4))*Power(z,2) + 11*(Power(x,2) + Power(y,2))*Power(z,4) + 6*Power(z,6)))/(Power(R,3)*Power(Power(x,2) + Power(z,2),2)*Power(Power(y,2) + Power(z,2),2)));
		} else if (a == 1 && b == 1 && c == 6) {
			return (x*y*(24 - (8*Power(x,2)*Power(y,4))/Power(Power(x,2) + Power(z,2),3) + (-20*Power(x,2)*Power(y,2) + 6*Power(y,4))/Power(Power(x,2) + Power(z,2),2) - (15*(x - y)*(x + y))/(Power(x,2) + Power(z,2)) - (8*Power(x,4)*Power(y,2))/Power(Power(y,2) + Power(z,2),3) + (6*Power(x,4) - 20*Power(x,2)*Power(y,2))/Power(Power(y,2) + Power(z,2),2) + (15*(x - y)*(x + y))/(Power(y,2) + Power(z,2))))/Power(R,5);
		} else if (a == 1 && b == 1 && c == 7) {
			return (3*x*y*z*(Power(x,2)*Power(y,2)*Power(Power(x,2) + Power(y,2),2)*(8*Power(x,8) + 12*Power(x,6)*Power(y,2) + 3*Power(x,4)*Power(y,4) + 12*Power(x,2)*Power(y,6) + 8*Power(y,8)) - Power(Power(x,2) + Power(y,2),3)*(8*Power(x,8) - 56*Power(x,6)*Power(y,2) - 3*Power(x,4)*Power(y,4) - 56*Power(x,2)*Power(y,6) + 8*Power(y,8))*Power(z,2) - Power(Power(x,2) + Power(y,2),2)*(60*Power(x,8) - 133*Power(x,6)*Power(y,2) - 177*Power(x,4)*Power(y,4) - 133*Power(x,2)*Power(y,6) + 60*Power(y,8))*Power(z,4) - 3*(Power(x,2) + Power(y,2))*(65*Power(x,8) - 34*Power(x,6)*Power(y,2) - 150*Power(x,4)*Power(y,4) - 34*Power(x,2)*Power(y,6) + 65*Power(y,8))*Power(z,6) - (345*Power(x,8) + 247*Power(x,6)*Power(y,2) - 156*Power(x,4)*Power(y,4) + 247*Power(x,2)*Power(y,6) + 345*Power(y,8))*Power(z,8) - 5*(Power(x,2) + Power(y,2))*(77*Power(x,4) + 10*Power(x,2)*Power(y,2) + 77*Power(y,4))*Power(z,10) - (303*Power(x,4) + 430*Power(x,2)*Power(y,2) + 303*Power(y,4))*Power(z,12) - 160*(Power(x,2) + Power(y,2))*Power(z,14) - 40*Power(z,16)))/(Power(R,7)*Power(Power(x,2) + Power(z,2),4)*Power(Power(y,2) + Power(z,2),4));
		} else if (a == 1 && b == 2 && c == 2) {
			return ArcTanh(x/R);
		} else if (a == 1 && b == 2 && c == 3) {
			return -((x*z)/(R*Power(y,2) + R*Power(z,2)));
		} else if (a == 1 && b == 2 && c == 4) {
			return (x*(-(Power(y,2)*(Power(x,2) + Power(y,2))) + (Power(x,2) + Power(y,2))*Power(z,2) + 2*Power(z,4)))/ (Power(R,3)*Power(Power(y,2) + Power(z,2),2));
		} else if (a == 1 && b == 2 && c == 5) {
			return (x*z*(Power(x,4)*(6*Power(y,2) - 2*Power(z,2)) + 5*Power(x,2)*(3*Power(y,2) - Power(z,2))*(Power(y,2) + Power(z,2)) + 3*(3*Power(y,2) - 2*Power(z,2))*Power(Power(y,2) + Power(z,2),2)))/ (Power(R,5)*Power(Power(y,2) + Power(z,2),3));
		} else if (a == 1 && b == 2 && c == 6) {
			return (3*x*(2*Power(x,6)*(Power(y,4) - 6*Power(y,2)*Power(z,2) + Power(z,4)) + 2*Power(x,2)*Power(Power(y,2) + Power(z,2),2)*(4*Power(y,4) - 27*Power(y,2)*Power(z,2) + 4*Power(z,4)) + Power(Power(y,2) + Power(z,2),3)*(3*Power(y,4) - 24*Power(y,2)*Power(z,2) + 8*Power(z,4)) + 7*Power(x,4)*(Power(y,6) - 5*Power(y,4)*Power(z,2) - 5*Power(y,2)*Power(z,4) + Power(z,6))))/ (Power(R,7)*Power(Power(y,2) + Power(z,2),4));
		} else if (a == 1 && b == 3 && c == 4) {
			return (x*y*(2*Power(x,4)*(Power(y,2) - 3*Power(z,2)) + 5*Power(x,2)*(Power(y,2) - 3*Power(z,2))*(Power(y,2) + Power(z,2)) + 3*(Power(y,2) - 4*Power(z,2))*Power(Power(y,2) + Power(z,2),2)))/ (Power(R,5)*Power(Power(y,2) + Power(z,2),3));
		} else if (a == 2 && b == 2 && c == 2) {
			return 1/R;
		} else if (a == 2 && b == 2 && c == 3) {
			return -(z/Power(R,3)); 
		} else if (a == 2 && b == 2 && c == 4) { 
			return -((Power(x,2) + Power(y,2) - 2*Power(z,2))/Power(R,5)); 
		} else if (a == 2 && b == 2 && c == 5) { 
			return (9*(Power(x,2) + Power(y,2))*z - 6*Power(z,3))/Power(R,7);
		} else if (a == 2 && b == 2 && c == 6) {
			return (9*Power(Power(x,2) + Power(y,2),2) - 72*(Power(x,2) + Power(y,2))*Power(z,2) + 24*Power(z,4))/Power(R,9);
		} else if (a == 2 && b == 3 && c == 3) {
			return (3*y*z)/Power(R,5);
		} else if (a == 2 && b == 3 && c == 4) {
			return (3*y*(Power(x,2) + Power(y,2) - 4*Power(z,2)))/Power(R,7);
		} else if (a == 2 && b == 3 && c == 5) {
			return (-45*y*(Power(x,2) + Power(y,2))*z + 60*y*Power(z,3))/Power(R,9);
		} else if (a == 2 && b == 4 && c == 4) {
			return (3*(Power(x,4) - 4*Power(y,4) + 27*Power(y,2)*Power(z,2) - 4*Power(z,4) - 3*Power(x,2)*(Power(y,2) + Power(z,2))))/Power(R,9);
		} else if (a == 3 && b == 3 && c == 4) {
			return (-15*x*y*(Power(x,2) + Power(y,2) - 6*Power(z,2)))/Power(R,9);
		}

		std::cerr << "\nf(" << a << "," << b << "," << c << ",lx,ly,lz,i,j,k) is not implemented!\n" << std::endl;
		assert(0);
		return 0.0;
	}

	template <class T>
	inline T ftermwise(int a, int b, int c, T lx, T ly, T lz, int i, int j, int k, int n)
	{
		if (a > b) return ftermwise(b, a, c, ly, lx, lz, j, i, k, n);
		if (b > c) return ftermwise(a, c, b, lx, lz, ly, i, k, j, n);

		const T x = i*lx;
		const T y = j*ly;
		const T z = k*lz;
		const T R = std::sqrt(x*x+y*y+z*z);

		if (a == 0 && b == 0 && c == 1) {
			if (n == 0 && i != 0 && k != 0) return -R*x*x*z/8.;
			if (n == 1 && j != 0 && k != 0) return -R*y*y*z/8.;
			if (n == 2 && k != 0) return + R*z*z*z/12.;
			if (n == 3 && i != 0 && j != 0 && k != 0) return -x*y*z*z*ArcTan((x*y)/(R*z))/2.;
			if (n == 4 && i != 0 && j != 0) return -x*y*y*y*ArcTan((x*z)/(R*y))/6.;
			if (n == 5 && i != 0 && j != 0) return -x*y*x*x*ArcTan((y*z)/(R*x))/6.;
			if (n == 6 && i != 0 && k != 0) return x*z*y*y*ArcTanh(x/R)/2.;
			if (n == 7 && i != 0 && k != 0) return -x*z*z*z*ArcTanh(x/R)/6.;
			if (n == 8 && j != 0 && k != 0) return y*z*x*x*ArcTanh(y/R)/2.;
			if (n == 9 && j != 0 && k != 0) return -y*z*z*z*ArcTanh(y/R)/6.;
			if (n == 10 && i != 0) return -x*x*x*x*ArcTanh(z/R)/24.;
			if (n == 11 && i != 0 && j != 0) return x*x*y*y*ArcTanh(z/R)/4.;
			if (n == 12 && j != 0) return -y*y*y*y*ArcTanh(z/R)/24.;
			return 0;
		} else if (a == 0 && b == 0 && c == 2) {
			if (n == 0 && i != 0) return -R*x*x/6.;
			if (n == 1 && j != 0) return -R*y*y/6.;
			if (n == 2 && k != 0) return R*z*z/3.;
			if (n == 3 && i != 0 && j != 0 && j != 0) return -x*y*z*ArcTan((x*y)/(R*z));
			if (n == 4 && i != 0 && j != 0) return x*y*y*ArcTanh(x/R)/2.;
			if (n == 5 && i != 0 && k != 0) return -x*z*z*ArcTanh(x/R)/2.;
			if (n == 6 && i != 0 && j != 0) return x*x*y*ArcTanh(y/R)/2.;
			if (n == 7 && j != 0 && k != 0) return -y*z*z*ArcTanh(y/R)/2.;
			return 0;
		} else if (a == 0 && b == 1 && c == 1) {
			if (n == 0 && j != 0 && k != 0) return -R*y*z/3.;
			if (n == 1 && i != 0 && k != 0) return -x*z*z*ArcTan((x*y)/(R*z))/2.;
			if (n == 2 && i != 0 && j != 0) return -x*y*y*ArcTan((x*z)/(R*y))/2.;
			if (n == 3 && i != 0) return -x*x*x*ArcTan((y*z)/(R*x))/6.;
			if (n == 4 && i != 0 && j != 0 && k != 0) return x*y*z*ArcTanh(x/R);
			if (n == 5 && i != 0 && k != 0) return x*x*z*ArcTanh(y/R)/2.;
			if (n == 6 && k != 0) return -z*z*z*ArcTanh(y/R)/6.;
			if (n == 7 && i != 0 && j != 0) return x*x*y*ArcTanh(z/R)/2.;
			if (n == 8 && j != 0) return -y*y*y*ArcTanh(z/R)/6.;
			return 0;
		}

		std::cerr << "\nftermwise(" << a << "," << b << "," << c << ",lx,ly,lz,i,j,k,n) is not implemented!\n" << std::endl;
		assert(0);
		return 0.0;
	}

	template <class T>
	inline T fseries(int a, int b, int c, int o, int p, int q, T lx, T ly, T lz, int i, int j, int k, int position)
	{
		assert(o != 0 && !(p == 0 && q != 0));

		T xprefactor = 1;
		if (position == SERIES_POSITION_CENTER) xprefactor *= 2*Power(lx, o)/Factorial(o);
		if (position == SERIES_POSITION_BORDER) xprefactor *= (Power(2, o)-2)*Power(-lx, o)/Factorial(o);
		if (position == SERIES_POSITION_BORDER) i++;

		if (q != 0) {
			return xprefactor*2*Power(ly, p)/Factorial(p)*2*Power(lz, q)/Factorial(q)*f(a+o, b+p, c+q, lx, ly, lz, i, j, k);
		}

		if (p != 0) {
			T sum1 = 0;
			for (int s = -1; s <= 1; s++) sum1 += (3*s*s-2)*f(a+o, b+p, c+q, lx, ly, lz, i, j, k+s);
			return xprefactor*2*Power(ly, p)/Factorial(p)*sum1;
		}

		T sum1 = 0;
		T sum2 = 0;
		for (int r = -1; r <= 1; r++) {
			sum2 = 0;
			for (int s = -1; s <= 1; s++) sum2 += (3*s*s-2)*f(a+o, b+p, c+q, lx, ly, lz, i, j+r, k+s);
			sum1 += (3*r*r-2)*sum2;
		}
		return xprefactor*sum1;
	}

	template <class T>
	inline T sumf_large_x(int o, int p, int q, T lx, T ly, T lz, int i, int j, int k)
	{
		if (p > q) return sumf_large_x(o, q, p, lx, lz, ly, i, k, j);
		const T x = i*lx;
		const T y = j*ly;
		const T z = k*lz;
		double sum = 0;

		if (o == 0 && p == 0 && q == 2) {
			sum += (Power(lx,2)*Power(ly,2)*Power(lz,2))/(Power(lx,2)*x - Power(x,3));
			sum += (Power(lx,2)*Power(ly,2)*Power(lz,2)*(Power(lx,4) - 3*Power(lx,2)*Power(x,2) + 6*Power(x,4))*(Power(ly,2) + 3*(Power(lz,2) + 2*Power(y,2) + 6*Power(z,2))))/ (24.*Power(-(Power(lx,2)*x) + Power(x,3),3));
			sum += -(Power(lx,2)*Power(ly,2)*Power(lz,2)*(Power(lx,8) - 5*Power(lx,6)*Power(x,2) + 10*Power(lx,4)*Power(x,4) - 5*Power(lx,2)*Power(x,6) + 15*Power(x,8))* (2*Power(ly,4) + 5*Power(ly,2)*(Power(lz,2) + 6*(Power(y,2) + Power(z,2))) + 10*(Power(lz,4) + 3*Power(lz,2)*(Power(y,2) + 5*Power(z,2)) + 3*(Power(y,2) + Power(z,2))*(Power(y,2) + 5*Power(z,2)))))/(240.*Power(-(Power(lx,2)*x) + Power(x,3),5));
			return sum;
		}
		if (o == 0 && p == 1 && q == 1) {
			sum += (Power(lx,2)*Power(ly,2)*Power(lz,2)*(Power(lx,4) - 3*Power(lx,2)*Power(x,2) + 6*Power(x,4))*y*z)/(2.*Power(-(Power(lx,2)*x) + Power(x,3),3));
			sum += -(Power(lx,2)*Power(ly,2)*Power(lz,2)*(Power(lx,8) - 5*Power(lx,6)*Power(x,2) + 10*Power(lx,4)*Power(x,4) - 5*Power(lx,2)*Power(x,6) + 15*Power(x,8))*y*z* (Power(ly,2) + Power(lz,2) + 2*(Power(y,2) + Power(z,2))))/(4.*Power(-(Power(lx,2)*x) + Power(x,3),5));
			sum += (5*Power(lx,2)*Power(ly,2)*Power(lz,2)*(Power(lx,12) - 7*Power(lx,10)*Power(x,2) + 21*Power(lx,8)*Power(x,4) - 35*Power(lx,6)*Power(x,6) + 42*Power(lx,4)*Power(x,8) + 14*Power(lx,2)*Power(x,10) + 28*Power(x,12))*y*z*(2*Power(ly,4) + Power(ly,2)*(3*Power(lz,2) + 10*Power(y,2) + 6*Power(z,2)) + 2*(Power(lz,4) + 3*Power(Power(y,2) + Power(z,2),2) + Power(lz,2)*(3*Power(y,2) + 5*Power(z,2)))))/(64.*Power(-(Power(lx,2)*x) + Power(x,3),7));
			return sum;
		}

		std::cerr << "\nsumf_large_x(" << o << "," << p << "," << o << ",lx,ly,lz,i,j,k) is not implemented!\n" << std::endl;
		assert(0);
		return 0.0;
	}

	template <class T>
	inline T sumf(int o, int p, int q, T ly, T lz, int i, int j, int k)
	{
		assert(ly >= 1. && lz >= ly);

		T lx = 1;

		double er = expansionradius;
		if ((o == 0 && p == 0 && q == 1) || (o == 0 && p == 1 && q == 0) || (o == 1 && p == 0 && q == 0)) er = expansionradius_oersted;
		if ((o == 0 && p == 0 && q == 2) || (o == 0 && p == 2 && q == 0) || (o == 2 && p == 0 && q == 0)) er = expansionradius_diag;
		if ((o == 0 && p == 1 && q == 1) || (o == 1 && p == 0 && q == 1) || (o == 1 && p == 1 && q == 0)) er = expansionradius_nondiag;
		const T Rsqr = std::min((i+1)*(i+1), std::min((i-1)*(i-1), i*i))*lx*lx+ std::min((j+1)*(j+1), std::min((j-1)*(j-1), j*j))*ly*ly+ std::min((k+1)*(k+1), std::min((k-1)*(k-1), k*k))*lz*lz;
		int expand = 0;
		if (lx*lx < Rsqr/(er*er)) expand++;
		if (ly*ly < Rsqr/(er*er)) expand++;
		if (lz*lz < Rsqr/(er*er)) expand++;

		double er2 = 6;

		if ((abs(j)+1)*(abs(j)+1)*ly*ly < Rsqr/(er2*er2) && (abs(k)+1)*(abs(k)+1)*lz*lz < Rsqr/(er2*er2) && ( (o == 0 && p == 2 && q == 0) || (o == 0 && p == 0 && q == 2) || (o == 0 && p == 1 && q == 1) )) {
			return sumf_large_x(o, p, q, lx, ly, lz, i, j, k);
		}

		if ((abs(i)+1)*(abs(i)+1)*lx*lx < Rsqr/(er2*er2) && (abs(k)+1)*(abs(k)+1)*lz*lz < Rsqr/(er2*er2) && ( (o == 2 && p == 0 && q == 0) || (o == 0 && p == 0 && q == 2) || (o == 1 && p == 0 && q == 1) )) {
			return sumf_large_x(p, o, q, ly, lx, lz, j, i, k);
		}

		if ((abs(i)+1)*(abs(i)+1)*lx*lx < Rsqr/(er2*er2) && (abs(j)+1)*(abs(j)+1)*ly*ly < Rsqr/(er2*er2) && ( (o == 2 && p == 0 && q == 0) || (o == 0 && p == 2 && q == 0) || (o == 1 && p == 1 && q == 0) )) {
			return sumf_large_x(q, o, p, lz, lx, ly, k, i, j);
		}

		T sum = 0;

		if (expand == 3) {
			sum += fseries(o, p, q, 2, 2, 2, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
			sum += fseries(o, p, q, 4, 2, 2, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
			sum += fseries(o, p, q, 2, 4, 2, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
			sum += fseries(o, p, q, 2, 2, 4, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
		}

		else if (expand == 2) {
			if (i == 0 && j == 0) {
				sum += fseries(o, p, q, 2, 2, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 3, 2, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 4, 2, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 5, 2, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 6, 2, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 2, 4, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
			}
			else {
				sum += fseries(o, p, q, 2, 2, 0, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
				sum += fseries(o, p, q, 4, 2, 0, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
				sum += fseries(o, p, q, 2, 4, 0, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
			}
		}

		else if (expand == 1) {
			if (i == 0) {
				sum += fseries(o, p, q, 2, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 3, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 4, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 5, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 6, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
				sum += fseries(o, p, q, 7, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_BORDER);
			}
			else {
				sum += fseries(o, p, q, 2, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
				sum += fseries(o, p, q, 4, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
				sum += fseries(o, p, q, 6, 0, 0, lx, ly, lz, i, j, k, SERIES_POSITION_CENTER);
			}
		}

		else {
			T sum1 = 0;
			T sum2 = 0;
			T sum3 = 0;
			for (int n = 0; n < 13; n++) {
				sum1 = 0;
				for (int c = -1; c <= 1; c++) {
					sum2 = 0;
					for (int b = -1; b <= 1; b++) {
						sum3 = 0;
						for (int a = -1; a <= 1; a++) sum3 += (3*a*a-2)*ftermwise(o, p, q, lx, ly, lz, i+a, j+b, k+c, n);
						sum2 += (3*b*b-2)*sum3;
					}
					sum1 += (3*c*c-2)*sum2;
				}
				sum += sum1;
			}
		}

		return sum;
	}

	inline double PBC_Demag_1D(int a, int b, int c, double lx, double ly, double lz, int i, int j, int k, int nx)
	{
		if (b > c) return PBC_Demag_1D(a, c, b, lx, lz, ly, i, k, j, nx);

		const double Lx = lx*nx;
		const double x = i*lx-0.5*Lx;
		const double y = j*ly;
		const double z = k*lz;
		const double R = std::sqrt(x*x+y*y+z*z);

		if (a == 2 && b == 0 && c == 0) {
			return (x*(3072*Power(R,12) - 384*Power(Lx,2)*Power(R,8)*(2*Power(x,2) - 3*(Power(y,2) + Power(z,2))) + 56*Power(Lx,4)*Power(R,4)*(8*Power(x,4) - 40*Power(x,2)*(Power(y,2) + Power(z,2)) + 15*Power(Power(y,2) + Power(z,2),2)) + 31*Power(Lx,6)*(-16*Power(x,6) + 168*Power(x,4)*(Power(y,2) + Power(z,2)) - 210*Power(x,2)*Power(Power(y,2) + Power(z,2),2) + 35*Power(Power(y,2) + Power(z,2),3))))/(3072.*Lx*Power(R,15));
		} else if (a == 0 && b == 0 && c == 2) {
			if (j != 0 || k != 0) {
				return ((Power(Lx,2)*x*(Power(x,2) + Power(y,2) - 4*Power(z,2)))/(8.*Power(R,7)) + (Power(y,2)*(-Power(R,3) + Power(x,3) + x*Power(y,2)) + (Power(R,3) - x*(Power(x,2) + Power(y,2)))*Power(z,2) - 2*x*Power(z,4))/(Power(R,3)*Power(Power(y,2) + Power(z,2),2)) - (7*Power(Lx,4)*x*(4*Power(x,4) + Power(x,2)*(Power(y,2) - 41*Power(z,2)) - 3*(Power(y,2) - 6*Power(z,2))*(Power(y,2) + Power(z,2))))/(384.*Power(R,11)) + (31*Power(Lx,6)*x*(8*Power(x,6) - 15*Power(x,2)*(Power(y,2) - 15*Power(z,2))*(Power(y,2) + Power(z,2)) + 5*(Power(y,2) - 8*Power(z,2))*Power(Power(y,2) + Power(z,2),2) - 12*Power(x,4)*(Power(y,2) + 13*Power(z,2))))/(3072.*Power(R,15)))/Lx;
			} else {
				return (31*Power(Lx,6) - 28*Power(Lx,4)*Power(x,2) + 48*Power(Lx,2)*Power(x,4) - 192*Power(x,6))/(384.*Lx*Power(x,8));
			}
		} else if (a == 0 && b == 1 && c == 1) {
			if (j != 0 || k != 0) {
				return (y*z*(-640*Power(Lx,2)*Power(R,8)*x + 392*Power(Lx,4)*Power(R,4)*x*(2*Power(x,2) - Power(y,2) - Power(z,2)) - 31*Power(Lx,6)*x*(48*Power(x,4) - 80*Power(x,2)*(Power(y,2) + Power(z,2)) + 15*Power(Power(y,2) + Power(z,2),2)) + (1024*Power(R,12)*(2*Power(R,3) - x*(2*Power(x,2) + 3*(Power(y,2) + Power(z,2)))))/Power(Power(y,2) + Power(z,2),2)))/(1024.*Lx*Power(R,15));
			} else {
				return 0;
			}
		} else if (a == 1 && b == 0 && c == 1) {
			return (z*(3072*Power(R,12) - 384*Power(Lx,2)*Power(R,8)*(4*Power(x,2) - Power(y,2) - Power(z,2)) + 168*Power(Lx,4)*Power(R,4)*(8*Power(x,4) - 12*Power(x,2)*(Power(y,2) + Power(z,2)) + Power(Power(y,2) + Power(z,2),2)) + 31*Power(Lx,6)*(-64*Power(x,6) + 240*Power(x,4)*(Power(y,2) + Power(z,2)) - 120*Power(x,2)*Power(Power(y,2) + Power(z,2),2) + 5*Power(Power(y,2) + Power(z,2),3))))/(3072.*Lx*Power(R,15));
		}

		assert(0);
		return 0;
	}

	inline double PBC_Demag_2D(int a, int b, int c, double lx, double ly, double lz, int i, int j, int k, int nx, int ny)
	{
		if (a > b) return PBC_Demag_2D(b, a, c, ly, lx, lz, j, i, k, ny, nx);

		const double Lx = lx*nx;
		const double Ly = ly*ny;
		const double x = i*lx-0.5*Lx;
		const double y = j*ly-0.5*Ly;
		const double z = k*lz;
		const double R = std::sqrt(x*x+y*y+z*z);

		if (a == 0 && b == 2 && c == 0) {
			return ((y*(-20*Power(Ly,2)*(Power(y,2) - 3*Power(z,2))*Power(Power(y,2) + Power(z,2),2) + 240*Power(Power(y,2) + Power(z,2),4) + 7*Power(Ly,4)*(Power(y,4) - 10*Power(y,2)*Power(z,2) + 5*Power(z,4))))/ Power(Power(y,2) + Power(z,2),5) + (x*y*(-240*Power(Lx,2)*Power(R,4) - (1920*Power(R,8))/(Power(y,2) + Power(z,2)) - 50*Power(Lx,2)*Power(Ly,2)*(3*Power(x,2) - 4*Power(y,2) + 3*Power(z,2)) + 35*Power(Lx,4)*(4*Power(x,2) - 3*(Power(y,2) + Power(z,2))) + (80*Power(Ly,2)*Power(R,4)*(2*Power(x,4)*(Power(y,2) - 3*Power(z,2)) + 5*Power(x,2)*(Power(y,2) - 3*Power(z,2))*(Power(y,2) + Power(z,2)) + 3*(2*Power(y,2) - 3*Power(z,2))*Power(Power(y,2) + Power(z,2),2)))/Power(Power(y,2) + Power(z,2),3) - (7*Power(Ly,4)*(8*Power(x,8)*(Power(y,4) - 10*Power(y,2)*Power(z,2) + 5*Power(z,4)) + 36*Power(x,6)*(Power(y,2) + Power(z,2))*(Power(y,4) - 10*Power(y,2)*Power(z,2) + 5*Power(z,4)) + 5*Power(Power(y,2) + Power(z,2),4)*(8*Power(y,4) - 40*Power(y,2)*Power(z,2) + 15*Power(z,4)) + 10*Power(x,2)*Power(Power(y,2) + Power(z,2),3)*(4*Power(y,4) - 55*Power(y,2)*Power(z,2) + 25*Power(z,4)) + 63*Power(x,4)*(Power(y,8) - 8*Power(y,6)*Power(z,2) - 14*Power(y,4)*Power(z,4) + 5*Power(z,8))))/Power(Power(y,2) + Power(z,2),5)))/(8.*Power(R,9)))/(240.*Lx*Ly);
		} else if (a == 0 && b == 0 && c == 2) {
			return -(5*x*y*(-192*Power(Ly,2)*Power(R,4) + 7*Power(Lx,4)*(59*Power(x,2) - 53*Power(y,2)) + 7*Power(Ly,4)*(-53*Power(x,2) + 59*Power(y,2)) + 2*Power(Lx,2)*(-96*Power(R,4) + 5*Power(Ly,2)*(Power(x,2) + Power(y,2)))) - 30*(21*Power(Lx,4) + 10*Power(Lx,2)*Power(Ly,2) + 21*Power(Ly,4))*x*y*Power(z,2) + (896*Power(Lx,4)*Power(x,5)*(Power(R,9) - Power(y,9)))/Power(Power(x,2) + Power(z,2),5) - (224*Power(Lx,4)*Power(x,3)*(5*Power(R,9) + 18*Power(x,2)*Power(y,7) - 5*Power(y,9)))/Power(Power(x,2) + Power(z,2),4) + (8*Power(Lx,2)*x*(80*Power(R,4)*Power(x,2)*(-Power(R,5) + Power(y,5)) + 7*Power(Lx,2)*(5*Power(R,9) - 126*Power(x,4)*Power(y,5) + 90*Power(x,2)*Power(y,7) - 5*Power(y,9))))/ Power(Power(x,2) + Power(z,2),3) - (20*Power(Lx,2)*x*(-24*Power(R,9) + 21*Power(Lx,2)*Power(y,3)*(14*Power(x,4) - 21*Power(x,2)*Power(y,2) + 3*Power(y,4)) + Power(R,4)*(-80*Power(x,2)*Power(y,3) + 24*Power(y,5))))/Power(Power(x,2) + Power(z,2),2) - (15*x*(-128*Power(R,9) + 128*Power(R,8)*y + 80*Power(Lx,2)*Power(R,4)*y*(-Power(x,2) + Power(y,2)) + 49*Power(Lx,4)*y*(3*Power(x,4) - 10*Power(x,2)*Power(y,2) + 3*Power(y,4))))/ (Power(x,2) + Power(z,2)) + (896*Power(Ly,4)*(Power(R,9) - Power(x,9))*Power(y,5))/Power(Power(y,2) + Power(z,2),5) - (224*Power(Ly,4)*Power(y,3)*(5*Power(R,9) - 5*Power(x,9) + 18*Power(x,7)*Power(y,2)))/Power(Power(y,2) + Power(z,2),4) + (640*Power(Ly,2)*Power(R,4)*(-Power(R,5) + Power(x,5))*Power(y,3) + 56*Power(Ly,4)*y*(5*Power(R,9) - 5*Power(x,9) + 90*Power(x,7)*Power(y,2) - 126*Power(x,5)*Power(y,4)))/ Power(Power(y,2) + Power(z,2),3) - (20*Power(Ly,2)*y*(-24*Power(R,9) + 8*Power(R,4)*(3*Power(x,5) - 10*Power(x,3)*Power(y,2)) + 21*Power(Ly,2)*Power(x,3)*(3*Power(x,4) - 21*Power(x,2)*Power(y,2) + 14*Power(y,4))))/Power(Power(y,2) + Power(z,2),2) - (15*y*(-128*Power(R,9) + 128*Power(R,8)*x + 80*Power(Ly,2)*Power(R,4)*x*(x - y)*(x + y) + 49*Power(Ly,4)*x*(3*Power(x,4) - 10*Power(x,2)*Power(y,2) + 3*Power(y,4))))/ (Power(y,2) + Power(z,2)))/(1920.*Lx*Ly*Power(R,9));
		} else if (a == 1 && b == 1 && c == 0) {
			return (1920*Power(R,8) - 10*Power(Lx,2)*Power(Ly,2)*(4*Power(x,4) - 27*Power(x,2)*Power(y,2) + 4*Power(y,4) + 3*(Power(x,2) + Power(y,2))*Power(z,2) - Power(z,4)) + 7*Power(Ly,4)*(3*Power(x,4) + 8*Power(y,4) - 24*Power(y,2)*Power(z,2) + 3*Power(z,4) + 6*Power(x,2)*(-4*Power(y,2) + Power(z,2))) + 7*Power(Lx,4)*(8*Power(x,4) - 24*Power(x,2)*(Power(y,2) + Power(z,2)) + 3*Power(Power(y,2) + Power(z,2),2)) + 80*Power(R,4)*(Power(Ly,2)*(Power(x,2) - 2*Power(y,2) + Power(z,2)) + Power(Lx,2)*(-2*Power(x,2) + Power(y,2) + Power(z,2))))/(1920.*Lx*Ly*Power(R,9));
		} else if (a == 0 && b == 1 && c == 1) {
			return ((20*Power(Ly,2)*z*(-3*Power(y,2) + Power(z,2))*Power(Power(y,2) + Power(z,2),2) + 240*z*Power(Power(y,2) + Power(z,2),4) + 7*Power(Ly,4)*(5*Power(y,4)*z - 10*Power(y,2)*Power(z,3) + Power(z,5)))/Power(Power(y,2) + Power(z,2),5) + (x*z*(-240*Power(Lx,2)*Power(R,4) - 50*Power(Lx,2)*Power(Ly,2)*(Power(x,2) - 6*Power(y,2) + Power(z,2)) - (1920*Power(R,8))/(Power(y,2) + Power(z,2)) + 35*Power(Lx,4)*(4*Power(x,2) - 3*(Power(y,2) + Power(z,2))) + (80*Power(Ly,2)*Power(R,4)* (Power(x,4)*(6*Power(y,2) - 2*Power(z,2)) + 5*Power(x,2)*(3*Power(y,2) - Power(z,2))*(Power(y,2) + Power(z,2)) + 3*(4*Power(y,2) - Power(z,2))*Power(Power(y,2) + Power(z,2),2)))/ Power(Power(y,2) + Power(z,2),3) - (7*Power(Ly,4)*(15*Power(Power(y,2) + Power(z,2),4)*(8*Power(y,4) - 12*Power(y,2)*Power(z,2) + Power(z,4)) + 8*Power(x,8)*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4)) + 36*Power(x,6)*(Power(y,2) + Power(z,2))*(5*Power(y,4) - 10*Power(y,2)*Power(z,2) + Power(z,4)) + 10*Power(x,2)*Power(Power(y,2) + Power(z,2),3)*(26*Power(y,4) - 53*Power(y,2)*Power(z,2) + 5*Power(z,4)) + 63*Power(x,4)*(5*Power(y,8) - 14*Power(y,4)*Power(z,4) - 8*Power(y,2)*Power(z,6) + Power(z,8))))/Power(Power(y,2) + Power(z,2),5)))/(8.*Power(R,9)))/(240.*Lx*Ly);
		}
			
		assert(0);
		return 0;
	}

	template <class T>
	inline T I_T_subsample(int o, int p, int q, T ly, T lz, int i, int j, int k, int sub_y, int sub_z)
	{
		assert(ly >= 1. && lz >= 1.);
		if (ly > lz) return I_T_subsample(o, q, p, lz, ly, i, k, j, sub_z, sub_y);

		if (sub_y > 0)
		{
			if (j != 0) return 2*I_T_subsample(o, p, q, ly/2.0, lz, i, 2*j, k, sub_y-1, sub_z)+I_T_subsample(o, p, q, ly/2.0, lz, i, 2*j-1, k, sub_y-1, sub_z)+I_T_subsample(o, p, q, ly/2.0, lz, i, 2*j+1, k, sub_y-1, sub_z);

			if (p % 2 == 0) return 2*I_T_subsample(o, p, q, ly/2.0, lz, i, 0, k, sub_y-1, sub_z)+2*I_T_subsample(o, p, q, ly/2.0, lz, i, 1, k, sub_y-1, sub_z);

			return 0;
		}

		if (sub_z > 0)
		{
			if (k != 0) return 2*I_T_subsample(o, p, q, ly, lz/2.0, i, j, 2*k, sub_y, sub_z-1)+I_T_subsample(o, p, q, ly, lz/2.0, i, j, 2*k-1, sub_y, sub_z-1)+I_T_subsample(o, p, q, ly, lz/2.0, i, j, 2*k+1, sub_y, sub_z-1);

			if (q % 2 == 0) return 2*I_T_subsample(o, p, q, ly, lz/2.0, i, j, 0, sub_y, sub_z-1)+2*I_T_subsample(o, p, q, ly, lz/2.0, i, j, 1, sub_y, sub_z-1);

			return 0;
		}

		return sumf(o, p, q, ly, lz, i, j, k);
	}

	template <class T>
	inline T I_T(int o, int p, int q, T lxp, T lyp, T lzp, int i, int j, int k)
	{
		if (lxp > lyp) return I_T(p, o, q, lyp, lxp, lzp, j, i, k);
		if (lyp > lzp) return I_T(o, q, p, lxp, lzp, lyp, i, k, j);

		if (o % 2 == 1 && i == 0) return 0;
		if (p % 2 == 1 && j == 0) return 0;
		if (q % 2 == 1 && k == 0) return 0;

		const T prefactor = Power(lxp, 5-o-p-q)/(4*PI*lxp*lyp*lzp);

		T lz = lzp/lxp;
		T ly = lyp/lxp;

/*
		T lx = 1;

		double er = expansionradius;
		if ((o == 0 && p == 0 && q == 1) || (o == 0 && p == 1 && q == 0) || (o == 1 && p == 0 && q == 0)) er = expansionradius_oersted;
		if ((o == 0 && p == 0 && q == 2) || (o == 0 && p == 2 && q == 0) || (o == 2 && p == 0 && q == 0)) er = expansionradius_diag;
		if ((o == 0 && p == 1 && q == 1) || (o == 1 && p == 0 && q == 1) || (o == 1 && p == 1 && q == 0)) er = expansionradius_nondiag;
		const T Rsqr = (std::min((i+1)*(i+1), std::min((i-1)*(i-1), i*i))*lx*lx
		              + std::min((j+1)*(j+1), std::min((j-1)*(j-1), j*j))*ly*ly
		              + std::min((k+1)*(k+1), std::min((k-1)*(k-1), k*k))*lz*lz)/(er*er);*/

		int sub_y = 0;
		//while (ly/Power(2, sub_y) > max_aspect_ratio && Rsqr <= ly*ly/Power(4, sub_y)) sub_y++;
		//if (Rsqr <= ly*ly/Power(4, sub_y)) sub_y = 0;

		int sub_z = 0;
		//while (lz/Power(2, sub_z) > max_aspect_ratio && Rsqr <= lz*lz/Power(4, sub_z)) sub_z++;
		//if (Rsqr <= lz*lz/Power(4, sub_z)) sub_z = 0;

		return prefactor*I_T_subsample(o, p, q, ly, lz, i, j, k, sub_y, sub_z);
	}

} // namespace

#endif
