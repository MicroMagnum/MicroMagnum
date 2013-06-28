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

#include <nvector/nvector_serial.h>
#include "matrix/matty.h"
#include "diffeq.h"

#define T0    RCONST(0.0)      /* initial time           */

class Cvode {

  public:
    /*
     * @param newton_method   Use Newton iteration or default: functional.
     *                        Functional is faster and Newton more stable.
     */
    Cvode(DiffEq &diff, double abstol, double reltol, bool newton_method);
    virtual ~Cvode();
    void evolve(double t, const double Tmax);

  private:
    static int callf(realtype t, N_Vector y, N_Vector ydot, void *user_data);

    N_Vector _Ny;
    double _reltol, _abstol;
    int _size;
    DiffEq& _diff;
    void *_cvode_mem;

};
#endif
