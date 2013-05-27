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

/*
 * Cvode constructor
 */
Cvode::Cvode(DiffEq &diff, double abstol, double reltol)
  : _Ny(), _diff(diff), _abstol(abstol), _reltol(reltol)
{
  _size = _diff.size();
  _Ny = N_VNew_Serial(_size);
  _diff.getN_Vector(_diff.getY(), _Ny);

  /*
   * CVode initialisation
   */
  _cvode_mem = NULL;

  _cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  assert( _cvode_mem != NULL);

  assert( CVodeSetUserData (_cvode_mem, &_diff)           == 0);
  assert( CVodeInit        (_cvode_mem, callf, T0, _Ny)   == 0);
  assert( CVodeSStolerances(_cvode_mem, _reltol, _abstol) == 0);
  assert( CVDense          (_cvode_mem, _size)            == 0);

}

Cvode::~Cvode()
{
  N_VDestroy_Serial(_Ny);
  CVodeFree(&_cvode_mem);
}

/*
 * calculate next step
 */
void Cvode::evolve(double t, const double step) 
{
  N_Vector yout;
  const double Tmax = t + step;

  yout = NULL;
  yout = N_VNew_Serial(_size);
  assert(yout != NULL);


  /* call CVode */
  assert( CVode(_cvode_mem, Tmax, yout, &t, CV_NORMAL) == 0);

  assert(yout != NULL);

  _diff.saveStateC(Tmax, yout);

  N_VDestroy_Serial(yout);
}


int Cvode::callf(realtype t, N_Vector Ny, N_Vector Nydot, void *user_data)
{
  DiffEq* ode = (DiffEq*) user_data;

  ode->diffN(Ny, Nydot, t);

  return(0);
}
