#!/usr/bin/python
from magnum import *
import unittest

try:
  import Image
  skip = False
except ImportError: # if python imaging lib is not installed
  skip = True

if not skip:
  class ImageShapeTest(unittest.TestCase):
    def setUp(self):
      mesh = RectangularMesh((100, 100, 1), (1, 1, 1))
      isc = ImageShapeCreator("world/shape/image_test.png", mesh)
  
      self.shape0 = isc.pick("blue")  # bottom-left of img -> (0,0)
      self.shape1 = isc.pick("green") # bottom-right -> (99,0)
      self.shape2 = isc.pick("black") # top-left -> (0,99)
      self.shape3 = isc.pick("red")   # top-right -> (99,99)
  
    def test_2d(self):
      mesh = RectangularMesh((100, 100, 1), (1, 1, 1))
      self.assertEquals([ 0+ 0*100], self.shape0.getCellIndices(mesh))
      self.assertEquals([99+ 0*100], self.shape1.getCellIndices(mesh))
      self.assertEquals([ 0+99*100], self.shape2.getCellIndices(mesh))
      self.assertEquals([99+99*100], self.shape3.getCellIndices(mesh))

# start tests
import os
if __name__ == '__main__':
  os.chdir("../..")
  unittest.main()
