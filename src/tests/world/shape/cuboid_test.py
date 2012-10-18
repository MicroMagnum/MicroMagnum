#!/usr/bin/python
from magnum import *
import unittest

# Cuboid
class CuboidTest(unittest.TestCase):
  def setUp(self):
    self.cube = Cuboid((0,0,0), (1,1,1))
    self.mesh = RectangularMesh((10, 10, 10), (1.0, 1.0, 1.0))

  def test_isPointInside(self):
    self.assertTrue (self.cube.isPointInside(( 0.5,  0.5,  0.5)))
    self.assertFalse(self.cube.isPointInside((-0.5,  0.5,  0.5)))

# start tests
if __name__ == '__main__':
  unittest.main()
