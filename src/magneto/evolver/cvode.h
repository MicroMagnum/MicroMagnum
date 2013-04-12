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

#ifndef CVODEINT_H
#define CVODEINT_H

//#include "config.h"

//#include "matrix/matty.h"
//#include <vector>
//#include <cvode/cvode.h>
//#include <cvode/cvode_band.h>
#include <nvector/nvector_serial.h>
//#include <sundials/sundials_types.h>
//#include <sundials/sundials_math.h>
//#include <sundials/sundials_band.h>
#include "matrix/matty.h"
#include "diffeq.h"

#define Y1    RCONST(1.0)      /* initial y components */
#define Y2    RCONST(1.0)
#define Y3    RCONST(1.0)
#define T0    RCONST(0.0)      /* initial time           */
#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */


class Cvode {

  public:
    /**
     * My     VectorMatrix y
     * Mydot  VectorMatrix y.differentiate
     * abstol Absolute Toleranz
     */
    Cvode(DiffEq &diff);
    virtual ~Cvode();
    int cvodeTest();

  private:
    int check_flag(void *flagvalue, char *funcname, int opt);
    static int callf(realtype t, N_Vector y, N_Vector ydot, void *user_data);

    typedef struct {
      realtype a,b,c,d;
    } *UserData;

    VectorMatrix _My, _Mydot;
    N_Vector _Ny, _Nydot, _abstol;
    double _reltol;
    int _size;
    DiffEq& _diff; //TODO Referenz

};
#endif
