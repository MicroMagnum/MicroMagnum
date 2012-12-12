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

#include "config.h"
#include "cvode.h"
#include "Magneto.h"
#include "/usr/local/include/cvode/cvode.h" // TODO relative path
//extern "C" {
//  #include "/usr/local/include/nvector/nvector_parallel.h"
//}
#include "/usr/local/include/nvector/nvector_serial.h"
#include "/usr/local/include/cvode/cvode_band.h"

// TODO implementation
int cvode_test() {
  realtype t;
  N_Vector y;
  N_Vector ydot;

  int ans;
  return ans;
}
