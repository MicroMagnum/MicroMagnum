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
import unittest

class VectorFieldTest(unittest.TestCase):

    def test_interpolate(self):
        mesh0 = RectangularMesh((100, 100, 1), (1e-9, 1e-9, 6e-9))
        mesh1 = RectangularMesh(( 50,  50, 1), (2e-9, 2e-9, 6e-9))

        M = VectorField(mesh0)
        M.fill((10, 20, 30))

        M2 = M.interpolate(mesh1)
        M2_avg = M2.average()

        self.assertEquals(10, M2_avg[0])
        self.assertEquals(20, M2_avg[1])
        self.assertEquals(30, M2_avg[2])

    def test_findExtremum(self):
        mesh0 = RectangularMesh((100, 100, 1), (1e-9, 1e-9, 6e-9))
        mesh1 = RectangularMesh(( 50,  50, 1), (2e-9, 2e-9, 6e-9))

        M = VectorField(mesh0)
        M.fill((100,100,100))
        M.findExtremum(z_slice=0, component=0)

if __name__ == '__main__':
    unittest.main()
