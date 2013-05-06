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

from magnum import RectangularMesh, Field, VectorField, readOMF
import magnum.magneto as magneto
import math

import unittest

from magnum_tests.helpers import MyTestCase

class UniaxialAnisotropyTest(MyTestCase):

    def doTest(self, M_file, H_file, epsilon):
        # load ref
        M     = readOMF(M_file)
        H_ref = readOMF(H_file)
        H     = VectorField(M.mesh)

        axis = VectorField(M.mesh); axis.fill((1.0/math.sqrt(3.0), 1.0/math.sqrt(3.0), 1.0/math.sqrt(3.0)))
        k = Field(M.mesh); k.fill(520e3)
        Ms = Field(M.mesh); Ms.fill(8e5)

        # calculate
        magneto.uniaxial_anisotropy(axis, k, Ms, M, H)

        # compare
        self.assertVectorFieldEqual(H_ref, H, epsilon)

    def test_calculate_2d(self):
        self.doTest("ref/M1.omf", "ref/H1_unianiso.omf", 10.0)

    def __test_calculate_3d(self):
        print("TODO: UniaxialAnisotropyTest.test_calculate_3d")

    def test_parallel_and_orthogonal_anisotropy(self):
        mesh = RectangularMesh((40, 40, 40), (1e-9, 1e-9, 1e-9))
        k = Field(mesh); k.fill(520e3)
        Ms = Field(mesh); Ms.fill(8e5)

        def calc(axis_vec, M_vec):
            axis = VectorField(mesh); axis.fill(axis_vec)
            M = VectorField(mesh); M.fill(M_vec)
            H = VectorField(mesh); H.fill((0, 0, 0))
            E = magneto.uniaxial_anisotropy(axis, k, Ms, M, H) * mesh.cell_volume
            return H.average(), E

        # parallel cases
        E_ref = 0.0

        H, E = calc((1,0,0), (8e5,0,0))
        self.assertAlmostEqual(E_ref, E)

        H, E = calc((0,1,0), (0,-8e5,0))
        self.assertAlmostEqual(E_ref, E)

        H, E = calc((0,0,-1), (0,0,-1))
        self.assertAlmostEqual(E_ref, E)

        # orthogonal cases
        E_ref = k.average() * mesh.volume

        H, E = calc((0,0,1), (8e5,0,0))
        self.assertAlmostEqual(E_ref, E)
        for i in range(3): self.assertAlmostEqual(0, H[i])

        H, E = calc((1,0,0), (0,-8e5,0))
        self.assertAlmostEqual(E_ref, E)
        for i in range(3): self.assertAlmostEqual(0, H[i])

        H, E = calc((0,-1,0), (0,0,1))
        self.assertAlmostEqual(E_ref, E)
        for i in range(3): self.assertAlmostEqual(0, H[i])

class CubicAnisotropyTest(unittest.TestCase):

    def test_parallel_anisotropy(self):
        mesh = RectangularMesh((40, 40, 40), (1e-9, 1e-9, 1e-9))
        k = Field(mesh); k.fill(520e3)
        Ms = Field(mesh); Ms.fill(8e5)

        def calc(axis1_vec, axis2_vec, M_vec):
            axis1 = VectorField(mesh); axis1.fill(axis1_vec)
            axis2 = VectorField(mesh); axis2.fill(axis2_vec)
            M = VectorField(mesh); M.fill(M_vec)
            H = VectorField(mesh); H.fill((0,0,0))
            E = magneto.cubic_anisotropy(axis1, axis2, k, Ms, M, H) * mesh.cell_volume
            return H.average(), E

        # parallel cases
        E_ref = 0.0

        H, E = calc((1,0,0), (0,1,0), (8e5,0,0))
        self.assertEqual(E_ref, E)
        for i in range(3): self.assertEqual(0.0, H[i])

        H, E = calc((1,0,0), (0,-1,0), (0,-8e5,0))
        self.assertEqual(E_ref, E)
        for i in range(3): self.assertEqual(0, H[i])

        H, E = calc((0,-1,0), (0,0,-1), (0,0,8e5))
        self.assertAlmostEqual(E_ref, E)
        for i in range(3): self.assertAlmostEqual(0, H[i])

import sys
if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
