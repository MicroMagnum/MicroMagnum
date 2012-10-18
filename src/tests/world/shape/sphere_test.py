#!/usr/bin/python
from magnum import *
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

# start tests
if __name__ == '__main__':
  unittest.main()
