/*
 * Copyright 2012 by the Micromagnum authors.
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

#include <Python.h>

// System
#include <stdlib.h>
#include <iostream>
//#include <stdio.h>
//#include <math.h>

// CVode includes
#include "cvode.h"
#include <cvode/cvode.h>
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <nvector/nvector_serial.h>
//#include <cvode/cvode_band.h>
//#include <cvode/cvode_band.h>        /* use CVBAND linear solver */
//#include <cvode/cvode_diag.h>        /* use CVDIAG linear solver */
//#include <sundials/sundials_types.h>
//#include <sundials/sundials_math.h>
//#include <sundials/sundials_band.h>

// Magnum
//#include "config.h"
#include "Magneto.h"
//#include "matrix/matty.h"
//#include "Vector3d.h"


static void PrintOutput(realtype t, realtype y1, realtype y2, realtype y3)
{
  std::cout << "t = " << t << ", y1 = " << y1 << ", y2 = " << y2 << ", y3 = " << y3 << "\n";
}

static void Cvode::matrixTest(VectorMatrix mat)
{
  int dim_x = mat.dimX();
  int dim_y = mat.dimY();
  int dim_z = mat.dimZ();
	const int dim_xy = dim_x * dim_y;

  std::cout << "matrixTest size: " << mat.size() << "\n";
  std::cout << "matrixTest dimX: " << mat.dimX() << "\n";
  std::cout << "matrixTest dimY: " << mat.dimY() << "\n";
  std::cout << "matrixTest dimZ: " << mat.dimZ() << "\n";

  VectorMatrix::const_accessor Macc(mat);

	for (int z=0; z<dim_z; ++z) {
		for (int y=0; y<dim_y; ++y) {	
			for (int x=0; x<dim_x; ++x) {
				const int i = z*dim_xy + y*dim_x + x; // linear index of (x,y,z)
        std::cout << Macc.get(i);
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

void Cvode::one(int i) {
  std::cout << "one from c++\n";
}

int Cvode::cvodeTest() {
  one(437291);
  realtype t;
  N_Vector yout, y, ydot, abstol;
  void *cvode_mem;
  UserData user_data;
  int flag, ans;

  y = ydot = yout = NULL;
  cvode_mem = NULL;
  user_data = NULL;

  y = N_VNew_Serial(3);
  ydot = N_VNew_Serial(3);
  yout = N_VNew_Serial(3);
  abstol = N_VNew_Serial(3);

  Ith(abstol,1) = 0.00001;
  Ith(abstol,2) = 0.00001;
  Ith(abstol,3) = 0.00001;


  Ith(y,1) = Y1;
  Ith(y,2) = Y2;
  Ith(y,3) = Y3;

  /* Call CVodeCreate to create the solver memory and specify the 
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if (check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  flag = CVodeInit(cvode_mem, callf, T0, y);
  if (check_flag(&flag, "CVodeInit", 1)) return(1);

  /* Call CVodeSVtolerances to specify the scalar relative tolerance
   *    * and vector absolute tolerances */
  flag = CVodeSVtolerances(cvode_mem, 0.00001, abstol);
  if (check_flag(&flag, "CVodeSVtolerances", 1)) return(1);

  /* Call CVDense to specify the CVDENSE dense linear solver */
  flag = CVDense(cvode_mem, 3);
  if (check_flag(&flag, "CVDense", 1)) return(1);

  flag = CVode(cvode_mem, 2, yout, &t, CV_NORMAL);
  if(check_flag(&flag, "CVode", 1));

  PrintOutput(t, Ith(yout,1), Ith(yout,2), Ith(yout,3));

  return ans;
}

/*
 * f routine. Compute function f(t,y). 
 */

static int Cvode::callf(realtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  realtype y1, y2, y3;

  //VectorMatrix tvec = getVectorMatrix(y);
  //N_Vector Y = getN_Vector(tvec);

  y1 = Ith(y,1); y2 = Ith(y,2); y3 = Ith(y,3);

    /*
     yd1 = Ith(ydot,1) = RCONST(-0.04)*y1 + RCONST(1.0e4)*y2*y3;
     yd3 = Ith(ydot,3) = RCONST(3.0e7)*y2*y2;
     Ith(ydot,2) = -yd1 - yd3;
     */

  Ith(ydot,1) = RCONST(1);
  Ith(ydot,2) = t;
  Ith(ydot,3) = t * t;

  return(0);
}

matty::VectorMatrix Cvode::f(matty::VectorMatrix y)
{
  std::cout << "c++ VectorMatrix\n";
  return y;
}

static N_Vector Cvode::getN_Vector(matty::VectorMatrix vec)
{
  //Array a = vec.getArray(0,0);
  N_Vector nvec = N_VNew_Serial(3);
  //Ith(nvec,1) = vec[0];
  //Ith(nvec,2) = vec[1];
  //Ith(nvec,3) = vec[2];

  return nvec;
}

static VectorMatrix Cvode::getVectorMatrix(N_Vector vec)
{
  VectorMatrix nvec;

  return nvec;
}

/*
 * Check function return value...
 * opt == 0 means SUNDIALS function allocates memory so check if
 *          returned NULL pointer
 * opt == 1 means SUNDIALS function returns a flag so check if
 *          flag >= 0
 * opt == 2 means function allocates memory so check if returned
 *          NULL pointer 
 */

static int Cvode::check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
        funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
          funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
        funcname);
    return(1); }

  return(0);
}

