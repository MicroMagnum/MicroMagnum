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

import unittest

from magnum import RectangularMesh, Field, VectorField
import magnum.magneto as magneto


class ScaledAbsMaxTest(unittest.TestCase):

    def test_1(self):
        mesh = RectangularMesh((10, 10, 10), (1e-9, 1e-9, 1e-9))

        M = VectorField(mesh); M.fill((-8e5, 0, 0))
        Ms = Field(mesh); Ms.fill(8e5)

        sam = magneto.scaled_abs_max(M, Ms)
        self.assertEqual(1.0, sam)

    def test_2(self):
        mesh = RectangularMesh((10, 10, 10), (1e-9, 1e-9, 1e-9))

        M = VectorField(mesh); M.fill((-8e5, 0, 0))
        Ms = Field(mesh); Ms.fill(8e5)
        Ms.set(0, 0)
        Ms.set(0, 8e5)
        self.assertFalse(Ms.isUniform())

        sam = magneto.scaled_abs_max(M, Ms)
        self.assertEqual(1.0, sam)

if __name__ == '__main__':
    unittest.main()
