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
%include "config.h"

#ifdef HAVE_CVODE
  %module(directors="1") cvode_director
#endif

%{
#include "evolver/runge_kutta.h"

#ifdef HAVE_CVODE
  #include "evolver/diffeq.h"
  #include "evolver/cvode.h"
#endif
%}

struct ButcherTableau
{
  /*
     a0 |
     a1 | b10
     a2 | b20 b21
     a3 | b30 b31 b32
     a4 | b40 b41 b42 b43
     a5 | b50 b51 b52 b53 b54 
     ---+-------------------------
        |  c0  c1  c2  c3  c4  c5
        | ec0 ec1 ec2 ec3 ec4 ec5
  */

  ButcherTableau(int num_steps);
  ~ButcherTableau();

  void   setA (int i, double v);
  double getA (int i);
  void   setB (int i, int j, double v);
  double getB (int i, int j);
  void   setC (int i, double v);
  double getC (int i);
  void   setEC(int i, double v);
  double getEC(int i);

  int getNumSteps();
};

void rk_prepare_step(
  int step,
  double h,
  ButcherTableau &tab,

  const VectorMatrix &k0,
  const VectorMatrix &k1,
  const VectorMatrix &k2,
  const VectorMatrix &k3,
  const VectorMatrix &k4,
  const VectorMatrix &k5,

  const VectorMatrix &y,
  VectorMatrix &ytmp
);

void rk_combine_result(
  double h,
  ButcherTableau &tab,

  const VectorMatrix &k0,
  const VectorMatrix &k1,
  const VectorMatrix &k2,
  const VectorMatrix &k3,
  const VectorMatrix &k4,
  const VectorMatrix &k5,

  VectorMatrix &y,
  VectorMatrix &y_error
);


double rk_adjust_stepsize(int order, double h, double eps_abs, double eps_rel, const VectorMatrix &y, const VectorMatrix &y_error);

#ifdef HAVE_CVODE
  %feature("director") DiffEq;
  %include "evolver/cvode.h";
  %include "evolver/diffeq.h";
#endif
