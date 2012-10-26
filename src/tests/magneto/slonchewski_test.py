#!/usr/bin/python

# Copyright 2012 by the Micromagnum authors.
#
# This file is part of MicroMagnum.
# 
# MicroMagnum is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# MicroMagnum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.

from magnum import *
import magnum.magneto as magneto

import unittest

# void fdm_slonchewski(
# 	int dim_x, int dim_y, int dim_z,
# 	double delta_x, double delta_y, double delta_z,
# 	double a_j,
# 	const VectorMatrix &p, // spin polarization
# 	const ScalarMatrix &Ms,
# 	const ScalarMatrix &alpha,
# 	const VectorMatrix &M,
# 	VectorMatrix &dM
# );

class FDMSlonchewskiTest(unittest.TestCase):

  def setup_stuff(self, dim_x, dim_y, dim_z, uniform):
    # self.dim_x, self.dim_y, self.dim_z = dim_x, dim_y, dim_z
    # self.delta_x, self.delta_y, self.delta_z = 3e-9, 3e-9, 3e-9
    # self.P = 1.0
    # self.xi = 0.02
    # self.Jx, self.Jy, self.Jz = 1e9, -1e9, 0.0
    # self.mesh = RectangularMesh((self.dim_x, self.dim_y, self.dim_z), (self.delta_x, self.delta_y, self.delta_z))
    # self.M = VectorField(self.mesh); self.M.fill((8e5,0,0))
    # self.Ms = Field(self.mesh); self.Ms.fill(8e5)
    # self.alpha = Field(self.mesh); self.alpha.fill(0.01)
    # self.dM = VectorField(self.mesh); self.dM.fill((0.0, 0.0, 0.0))

    # if uniform:
    #   self.assertTrue(self.Ms.isUniform())
    #   self.assertTrue(self.alpha.isUniform())
    # else:
    #   self.Ms.setAt(0, self.Ms.getAt(0))
    #   self.alpha.setAt(0, self.alpha.getAt(0))
    #   self.assertFalse(self.Ms.isUniform())
    #   self.assertFalse(self.alpha.isUniform())
    pass

# start tests
import os
if __name__ == '__main__':
  os.chdir("../..")
  unittest.main()
