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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "config.h"

// according to Wikipedia :)
#define MY_PI 3.141592653589793238462643383279502884197169399

#define MU0 (4*MY_PI*1e-7)

// Plancks constant [kg*m^2/s]
#define H_BAR 1.05457162825e-34 

// electron charge [Coulomb] = [A*s]
#define ELECTRON_CHARGE 1.602176487e-19

// Bohr magneton [J/T]
#define MU_BOHR 9.2740154e-24

// Gyromagnetic ratio
#define GYROMAGNETIC_RATIO 2.211e5

// K Boltzmann
#define BOLTZMANN_CONSTANT 1.3806504e-23

#endif
