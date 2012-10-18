#!/usr/bin/python
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
