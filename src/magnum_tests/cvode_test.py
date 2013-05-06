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

from magnum import *
import magnum.magneto as m
from magnum.llgDiffEq import *
from math import pi, cos, sin

import unittest
import itertools

class CvodeTest(unittest.TestCase):

    def setUp(self):
        world = World(RectangularMesh((3,  3, 3), (  5e-9,    5e-9, 3.0e-9)), Body("all", Material.Py(alpha=0.02)))
        def state0(field, pos): 
            u = abs(pi*(pos[0]/field.mesh.size[0]-0.5)) / 2.0
            return 8e5 * cos(u), 8e5 * sin(u), 0
        self.solver = create_solver(world, [StrayField, ExchangeField], log=True, do_precess=False, evolver="rkf45", eps_abs=1e-4, eps_rel=1e-2)
        self.solver.state.M = state0
        self.solver.state.alpha = 0.5
        self.llg = LlgDiffEq(self.solver.state)
        self.cv = m.Cvode(self.llg)
            
    def test_getY(self):
        self.assertEqual(self.llg.getY(), self.solver.state.y)

    def test_diffX(self):
        y = self.solver.state.y
        ydot = y
        ydot2 = self.solver.state.differentiate()
        self.llg.diffX(y, ydot, 1)
        print("ydot:")
        self.llg.printVectorMatrix(ydot)
        print("ydot2:")
        self.llg.printVectorMatrix(ydot2)
        # TODO equal test
        #self.assertEqual(ydot, ydot2)
        

if __name__ == '__main__':
    unittest.main()

