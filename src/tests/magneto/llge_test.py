#!/usr/bin/python
from magnum import *
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
