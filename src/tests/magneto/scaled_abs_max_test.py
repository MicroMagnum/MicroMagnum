#!/usr/bin/python
from magnum import *
import magnum.magneto as magneto

import unittest

class ScaledAbsMaxTest(unittest.TestCase):

  def test_1(self):
    mesh = RectangularMesh((10,10,10), (1e-9,1e-9,1e-9))

    M = VectorField(mesh); M.fill((-8e5,0,0))
    Ms = Field(mesh); Ms.fill(8e5)

    sam = magneto.scaled_abs_max(M, Ms)
    self.assertEqual(1.0, sam)

  def test_2(self):
    mesh = RectangularMesh((10,10,10), (1e-9,1e-9,1e-9))

    M = VectorField(mesh); M.fill((-8e5,0,0))
    Ms = Field(mesh); Ms.fill(8e5)
    Ms.set(0, 0)
    Ms.set(0, 8e5)
    self.assertFalse(Ms.isUniform())

    sam = magneto.scaled_abs_max(M, Ms)
    self.assertEqual(1.0, sam)

if __name__ == '__main__':
  unittest.main()
