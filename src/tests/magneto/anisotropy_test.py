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

  def test_parallel_and_orthogonal_anisotropy(self):
    mesh = RectangularMesh((40, 40, 40), (1e-9, 1e-9, 1e-9))
    k = Field(mesh); k.fill(520e3)
    Ms = Field(mesh); Ms.fill(8e5)

    def calc(axis_vec, M_vec):
      axis = VectorField(mesh); axis.fill(axis_vec)
      M = VectorField(mesh); M.fill(M_vec)
      H = VectorField(mesh); H.fill((0,0,0))
      E = magneto.uniaxial_anisotropy(axis, k, Ms, M, H) * mesh.cell_volume
      return H.average(), E

    # parallel cases
    E_ref = 0.0

    H, E = calc((1,0,0), (8e5,0,0))
    self.assertAlmostEqual(E_ref, E)

    H, E = calc((0,1,0), (0,-8e5,0))
    self.assertAlmostEqual(E_ref, E)

    H, E = calc((0,0,-1), (0,0,-1))
    self.assertAlmostEqual(E_ref, E)
    
    # orthogonal cases
    E_ref = k.average() * mesh.cell_volume * mesh.total_nodes

    H, E = calc((0,0,1), (8e5,0,0))
    self.assertAlmostEqual(E_ref, E)
    self.assertAlmostEqual((0,0,0), H)

    H, E = calc((1,0,0), (0,-8e5,0))
    self.assertAlmostEqual(E_ref, E)
    self.assertAlmostEqual((0,0,0), H)

    H, E = calc((0,-1,0), (0,0,1))
    self.assertAlmostEqual(E_ref, E)
    self.assertAlmostEqual((0,0,0), H)

class CubicAnisotropyTest(unittest.TestCase):
  def test_parallel_and_orthogonal_anisotropy(self):
    self.fail("Need to implement testcase for cubic anisotropy.")

import os
if __name__ == '__main__':
  os.chdir("..")
  unittest.main()
