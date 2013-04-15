#!/usr/bin/python

# Copyright 2012, 2013 by the Micromagnum authors.
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

from __future__ import print_function

from magnum import *
import magnum.magneto as magneto

import unittest

# void fdm_zhangli(
#   int dim_x, int dim_y, int dim_z,
#   double delta_x, double delta_y, double delta_z,
#   bool do_precess,
#   double P, double xi, double Jx, double Jy, double Jz,
#   const ScalarMatrix &Ms,
#   const ScalarMatrix &alpha,
#   const VectorMatrix &M,
#   VectorMatrix &dM
# );

class FDMZhangLiTest(unittest.TestCase):

    def setup_stuff(self, dim_x, dim_y, dim_z, uniform):
        self.dim_x, self.dim_y, self.dim_z = dim_x, dim_y, dim_z
        self.delta_x, self.delta_y, self.delta_z = 3e-9, 3e-9, 3e-9
        self.mesh = RectangularMesh((self.dim_x, self.dim_y, self.dim_z), (self.delta_x, self.delta_y, self.delta_z))
        self.P = Field(self.mesh); self.P.fill(1.0)
        self.xi = Field(self.mesh); self.xi.fill(0.02)
        self.J = VectorField(self.mesh); self.J.fill((1e9, -1e9, 0.0))
        self.M = VectorField(self.mesh); self.M.fill((8e5,0,0))
        self.Ms = Field(self.mesh); self.Ms.fill(8e5)
        self.alpha = Field(self.mesh); self.alpha.fill(0.01)
        self.dM = VectorField(self.mesh); self.dM.fill((0.0, 0.0, 0.0))

        if uniform:
            self.assertTrue(self.Ms.isUniform())
            self.assertTrue(self.alpha.isUniform())
        else:
            x = self.Ms.get(0); self.Ms.set(0,-1); self.Ms.set(0,x)
            x = self.alpha.get(0); self.alpha.set(0,-1); self.alpha.set(0,x)
            self.assertFalse(self.Ms.isUniform())
            self.assertFalse(self.alpha.isUniform())

    def with_zero_gradient_test(self, dim_x, dim_y, dim_z, uniform):
        self.setup_stuff(dim_x, dim_y, dim_z, uniform)

        # No gradient in M -> no spin torque
        self.M.fill((0.0, 0.0, 0.0))
        magneto.fdm_zhangli(
          self.dim_x, self.dim_y, self.dim_z, self.delta_x, self.delta_y, self.delta_z, True,
          self.P, self.xi,
          self.Ms, self.alpha,
          self.J, self.M, self.dM
        )
        self.assertFloatTupleEq((0.0, 0.0, 0.0), self.dM.average(), epsilon=0.001) # spin torque is zero

    def with_vortex_test(self, dim_x, dim_y, dim_z, uniform):
        self.setup_stuff(dim_x, dim_y, dim_z, uniform)

        vortex_fn = vortex.magnetizationFunction(150e-9, 150e-9)
        for n in range(self.mesh.total_nodes):
            x, y, z = self.mesh.getPosition(n)
            self.M.set(n, vortex_fn(self.M, (x, y, z)))

        magneto.fdm_zhangli(
          self.dim_x, self.dim_y, self.dim_z, self.delta_x, self.delta_y, self.delta_z, True,
          self.P, self.xi,
          self.Ms, self.alpha,
          self.J, self.M, self.dM
        )

        ref_avg = (340104691300.73352, 337831145947.55255, 7.4024200439453129e-06)
        #print("Reference <dM>: ", ref_avg)
        #print("Actual    <dM>: ", self.dM.average())

        # epsilon must be relatively big...
        self.assertFloatTupleEq(self.dM.average(), ref_avg, epsilon=1000000.0)

    def assertFloatTupleEq(self, ref, actual, epsilon):
        self.assertTrue(len(ref) == len(actual))
        for a, b in zip(ref, actual):
            self.assertTrue(abs(a-b) < epsilon)

    def test_2d_zero_gradient(self): self.with_zero_gradient_test(100, 100, 1, True)
    def test_3d_zero_gradient(self): self.with_zero_gradient_test(100, 100, 16, True)
    def test_2d(self): self.with_vortex_test(100, 100, 1, True)
    def test_3d(self): self.with_vortex_test(100, 100, 20, True)

    def test_2d_zero_gradient_2(self): self.with_zero_gradient_test(100, 100, 1, False)
    def test_3d_zero_gradient_2(self): self.with_zero_gradient_test(100, 100, 16, False)
    def test_2d_2(self): self.with_vortex_test(100, 100, 1, False)
    def test_3d_2(self): self.with_vortex_test(100, 100, 20, False)

if __name__ == '__main__':
    unittest.main()
