#!/usr/bin/python
from magnum import *
import magnum.magneto as magneto
import math

import unittest
from .my_testcase import MyTestCase

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
    self.doTest("ref/M1.omf", "ref/H1_unianiso.omf", 10.0);

  def __test_calculate_3d(self):
    print("TODO: UniaxialAnisotropyTest.test_calculate_3d")

class CubicAnisotropyTest(unittest.TestCase):
  pass

import os
if __name__ == '__main__':
  os.chdir("..")
  unittest.main()
