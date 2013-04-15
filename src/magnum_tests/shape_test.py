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

from magnum import RectangularMesh, Sphere, Cylinder, Cuboid, Everywhere

import math
import unittest

class SphereTest(unittest.TestCase):

    def setUp(self):
        self.sphere = Sphere((1,1,1), 1)
        self.mesh = RectangularMesh((20, 20, 20), (0.1, 0.1, 0.1))

    def test_isPointInside(self):
        self.assertTrue (self.sphere.isPointInside((1.0, 1.0, 1.0)))
        self.assertTrue (self.sphere.isPointInside((1.9, 1.0, 1.0)))
        self.assertTrue (self.sphere.isPointInside((1.0, 0.1, 1.0)))
        self.assertTrue (self.sphere.isPointInside((1.0, 1.0, 1.9)))
        self.assertFalse(self.sphere.isPointInside((2.1, 1.0, 1.0)))
        self.assertFalse(self.sphere.isPointInside((1.9, 1.9, 1.0)))
        self.assertFalse(self.sphere.isPointInside((0.1, 0.1, 1.0)))

class CylinderTest(unittest.TestCase):

    def setUp(self):
        P1, P2, R = (0,0,0), (40,0,0), 10
        self.cyl1 = Cylinder(P1, P2, R)

    def test_isPointInside(self):
        f = 10.0 / math.sqrt(2.0)
        for P in [(10,0,0), (20,0,0), (30,0,0), (10,f-0.1,f-0.1), (10,-f+0.1,-f+0.1)]:
            self.assertTrue(self.cyl1.isPointInside(P))
        for P in [(-10,0,0), (50,0,0), (10,f+0.1,f+0.1), (10,-f-0.1,-f-0.1)]:
            self.assertFalse(self.cyl1.isPointInside(P))

class CuboidTest(unittest.TestCase):

    def setUp(self):
        self.cube = Cuboid((0,0,0), (1,1,1))
        self.mesh = RectangularMesh((10, 10, 10), (1.0, 1.0, 1.0))

    def test_isPointInside(self):
        self.assertTrue (self.cube.isPointInside(( 0.5,  0.5,  0.5)))
        self.assertFalse(self.cube.isPointInside((-0.5,  0.5,  0.5)))

class EverywhereTest(unittest.TestCase):

    def test_isPointInside(self):
        ew = Everywhere()
        self.assertTrue(ew.isPointInside(( 0.0,  0.0,  0.0)))
        self.assertTrue(ew.isPointInside((-5.0,  5.0,  0.0)))

    def test_getCellIndices(self):
        mesh = RectangularMesh((10,10,10),(1,1,1))
        ew = Everywhere()
        idx = ew.getCellIndices(mesh)
        self.assertEqual(1000, len(idx))

class CombinedShapeTest(unittest.TestCase):

    def setUp(self):
        self.cube1 = Cuboid((0,0,0), (1,1,1))
        self.cube2 = Cuboid((1,1,1), (2,2,2))
        self.cube3 = Cuboid((0.5,0.5,0.5), (1.5,1.5,1.5))

    def test_isPointInside(self):
        combo = self.cube1 | self.cube2
        self.assertTrue (combo.isPointInside((0.5, 0.5, 0.5)))
        self.assertTrue (combo.isPointInside((1.5, 1.5, 1.5)))
        self.assertFalse(combo.isPointInside((0.5, 1.0, 1.5)))

        combo = self.cube1 & self.cube3
        self.assertTrue (combo.isPointInside((0.75,0.75,0.75)))
        self.assertFalse(combo.isPointInside((0.3,0.3,0.3)))
        self.assertFalse(combo.isPointInside((1.3,1.3,1.3)))

    def test_getCellIndices(self):
        mesh = RectangularMesh((20, 20, 20), (0.1, 0.1, 0.1))

        combo = self.cube1 | self.cube2
        cells = combo.getCellIndices(mesh)
        self.assertEqual(2000, len(cells))

        combo = self.cube1 & self.cube2
        cells = combo.getCellIndices(mesh)
        self.assertEqual(0, len(cells))

if __name__ == '__main__':
    unittest.main()
