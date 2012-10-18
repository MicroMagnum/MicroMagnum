#!/usr/bin/python
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
