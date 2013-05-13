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

#define Ith(v,i)    NV_Ith_S(v,i)       /* Ith numbers components 1..NEQ */ // TODO remove

Cvode::Cvode(DiffEq &diff)
  : _Ny(), _diff(diff)
{
  _size = _diff.size();
  std::cout << "size: " << _size << "\n";

  _Ny = N_VNew_Serial(_size);

  _diff.getN_Vector(_diff.getY(), _Ny);
  //_diff.printN_Vector(_Ny);

  _reltol = 1e-4;
  _abstol = 1e1;
}

Cvode::~Cvode()
{
}

void Cvode::cvodeCalculate() 
{
  realtype t;
  N_Vector yout;
  void *cvode_mem;
  int flag;

  yout = NULL;
  cvode_mem = NULL;
  yout = N_VNew_Serial(_size);
  assert(yout != NULL);

  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  assert(cvode_mem != NULL);
  flag = CVodeSetUserData(cvode_mem, &_diff);
  assert(flag == 0);
  flag = CVodeInit(cvode_mem, callf, T0, _Ny);
  assert(flag == 0);
  flag = CVodeSStolerances(cvode_mem, _reltol, _abstol);
  assert(flag == 0);
  flag = CVDense(cvode_mem, _size);
  assert(flag == 0);

  _diff.printOutput(t,_Ny);

  flag = CVode(cvode_mem, Tmax, yout, &t, CV_NORMAL);
  assert(flag == 0);

  _diff.printOutput(t,yout);
  assert(yout != NULL);

  N_VDestroy_Serial(_Ny);
  CVodeFree(&cvode_mem);
  std::cout << "done\n";
}


int Cvode::callf(realtype t, N_Vector Ny, N_Vector Nydot, void *user_data)
{
  DiffEq* ode = (DiffEq*) user_data;

  ode->diffN(Ny, Nydot, t);
  //ode->printOutput(t,Nydot);

  return(0);
}
