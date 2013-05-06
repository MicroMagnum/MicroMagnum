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
#include <assert.h>

// CVode includes
#include "cvode.h"
#include <cvode/cvode.h>
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <nvector/nvector_serial.h>

// Magnum
//#include "config.h"
#include "Magneto.h"
#include "diffeq.h"

Cvode::Cvode(DiffEq &diff)
  : _Ny(), _diff(diff)
{
  _size = _diff.size();
  std::cout << "size: " << _size << "\n";

  _Ny = N_VNew_Serial(_size);

  _diff.getN_Vector(_diff.getY(), _Ny);
  //_diff.printN_Vector(_Ny);

  _reltol = 1e-60;
  _abstol = 4e8;
}

Cvode::~Cvode()
{
}

void Cvode::cvodeCalculate() 
{
  realtype t;
  N_Vector yout;
  void *cvode_mem;
  int flag, ans;

  yout = NULL;
  cvode_mem = NULL;
  yout = N_VNew_Serial(_size);

  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  assert(check_flag((void *)cvode_mem, (char *) "CVodeCreate", 0) == 0);

  flag = CVodeSetUserData(cvode_mem, &_diff);
  assert(check_flag(&flag, (char *) "CVodeSetUserData", 1) == 0);

  flag = CVodeInit(cvode_mem, callf, T0, _Ny);
  assert(check_flag(&flag, (char *) "CVodeInit", 1) == 0);

  flag = CVodeSStolerances(cvode_mem, _reltol, _abstol);
  assert(check_flag(&flag, (char *) "CVodeSVtolerances", 1) == 0);

  flag = CVDense(cvode_mem, 3);
  assert(check_flag(&flag, (char *) "CVDense", 1) == 0);

  std::cout << "cvode 1\n";

  flag = CVode(cvode_mem, Tmax, yout, &t, CV_NORMAL);
  assert(check_flag(&flag, (char *) "CVode", 1) == 0);
  std::cout << "cvode 2\n";

  _diff.printOutput(t,yout);
  assert(yout != NULL);
  std::cout << "cvode 3\n";

  CVodeFree(&cvode_mem);
  std::cout << "cvode 4\n";
  N_VDestroy_Serial(_Ny);
  std::cout << "cvode 5\n";
}


int Cvode::callf(realtype t, N_Vector Ny, N_Vector Nydot, void *user_data)
{
  DiffEq* ode = (DiffEq*) user_data;

  std::cout << "callf 1\n";
  ode->diffN(Ny, Nydot, t);
  std::cout << "callf 2\n";
  ode->printOutput(t,Nydot);
  std::cout << "callf 3\n";

  return(0);
}


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

