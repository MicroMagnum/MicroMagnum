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

from magnum import RectangularMesh, Field, VectorField
import magnum.magneto as magneto
import unittest

class LLGETest(unittest.TestCase):

    def test_llge(self):
        mesh = RectangularMesh((10,10,10), (1e-9,1e-9,1e-9))
        f1, f2, M, H, dM = Field(mesh), Field(mesh), VectorField(mesh), VectorField(mesh), VectorField(mesh)

        # dM = f1*MxH + f2*Mx(MxH)
        f1.fill(10)
        f2.fill(20)
        M.fill((5,10,15))
        H.fill((20,25,30))
        magneto.llge(f1, f2, M, H, dM)

        for idx in range(dM.size()):
            self.assertEqual(dM.get(idx), (-60750.0, -13500.0, 29250.0))

if __name__ == '__main__':
    unittest.main()
