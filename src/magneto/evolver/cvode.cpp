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
#include "diffeq.h"
//#include "matrix/matty.h"
//#include "Vector3d.h"

Cvode::Cvode(DiffEq &diff)
  : _Ny(), _Nydot(), _abstol(), _diff(diff)
{
  _size = _diff.size();
  std::cout << "size: " << _size << "\n";

  _Ny = N_VNew_Serial(_size);
  _Nydot = N_VNew_Serial(_size);
  _abstol = N_VNew_Serial(_size);

  _diff.getN_Vector(_diff.getY(), _Ny);
  _diff.printN_Vector(_Ny);

  _reltol = 0.1;

  for (int i=1; i<=_size; ++i)
  {
    Ith(_abstol,i) = 0.1;
  }
}

Cvode::~Cvode()
{
}

int Cvode::cvodeTest() 
{
  realtype t;
  N_Vector yout;
  void *cvode_mem;
  int flag, ans;

  yout = NULL;
  cvode_mem = NULL;
  //user_data = NULL;

  //y = N_VNew_Serial(3);
  //ydot = N_VNew_Serial(3);
  yout = N_VNew_Serial(_size);

  /* Call CVodeCreate to create the solver memory and specify the 
   * Backward Differentiation Formula and the use of a Newton iteration */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if (check_flag((void *)cvode_mem, (char *) "CVodeCreate", 0)) return(1);

  /* Set the pointer to user-defined data */
  flag = CVodeSetUserData(cvode_mem, &_diff);
  if(check_flag(&flag, (char *) "CVodeSetUserData", 1)) return(1);

  /* Call CVodeInit to initialize the integrator memory and specify the
   * user's right hand side function in y'=f(t,y), the inital time T0, and
   * the initial dependent variable vector y. */
  flag = CVodeInit(cvode_mem, callf, T0, _Ny);
  if (check_flag(&flag, (char *) "CVodeInit", 1)) return(1);

  /* Call CVodeSVtolerances to specify the scalar relative tolerance
   * and vector absolute tolerances */
  flag = CVodeSVtolerances(cvode_mem, _reltol, _abstol);
  if (check_flag(&flag, (char *) "CVodeSVtolerances", 1)) return(1);

  /* Call CVDense to specify the CVDENSE dense linear solver */
  flag = CVDense(cvode_mem, 3);
  if (check_flag(&flag, (char *) "CVDense", 1)) return(1);

  std::cout << "cvode 1\n";

  flag = CVode(cvode_mem, 1, yout, &t, CV_NORMAL);
  if(check_flag(&flag, (char *) "CVode", 1)) return(1);
  std::cout << "cvode 2\n";

  _diff.printOutput(t,yout);
  std::cout << "cvode 3\n";

  free(cvode_mem);
  return ans;
}

/*
 * f routine. Compute function f(t,y). 
 */

int Cvode::callf(realtype t, N_Vector Ny, N_Vector Nydot, void *user_data)
{
  DiffEq* ode = (DiffEq*) user_data;

  std::cout << "callf 1\n";
  ode->diffN(Ny, Nydot);
  std::cout << "callf 2\n";
  ode->printOutput(t,Nydot);
  std::cout << "callf 3\n";

  return(0);
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

int Cvode::check_flag(void *flagvalue, char *funcname, int opt)
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

