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

#define Y1    RCONST(1.0)      /* initial y components */
#define Y2    RCONST(1.0)
#define Y3    RCONST(1.0)
#define T0    RCONST(0.0)      /* initial time           */
#define Ith(v,i)    NV_Ith_S(v,i-1)       /* Ith numbers components 1..NEQ */


class Cvode {

  public:
    virtual int cvodeTest();

    virtual void one(int i);

    virtual matty::VectorMatrix f(matty::VectorMatrix y);

    static int callf(realtype t, N_Vector y, N_Vector ydot, void *user_data);

    static void matrixTest(VectorMatrix mat);

  private:
    static void getVectorMatrix(N_Vector vec, VectorMatrix& mat);

    static void getN_Vector(matty::VectorMatrix vec, N_Vector& nvec);

    static int check_flag(void *flagvalue, char *funcname, int opt);

    typedef struct {
      realtype a,b,c,d;
    } *UserData;

};
