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
import unittest

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

# start tests
if __name__ == '__main__':
  unittest.main()
