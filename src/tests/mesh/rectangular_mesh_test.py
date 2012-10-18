#!/usr/bin/python
from magnum import *
import unittest

# Body
class RectangularMeshTest(unittest.TestCase):

  def test_derived_quantities(self):
    mesh = RectangularMesh((10,10,10),(2,2,2))
    self.assertEqual(mesh.cell_volume, 8)
    self.assertEqual(mesh.volume, 8000)
    self.assertEqual(mesh.size, (20,20,20))
    self.assertEqual(mesh.delta, (2,2,2))
    self.assertEqual(mesh.num_nodes, (10,10,10))
    self.assertEqual(mesh.total_nodes, 1000)
    self.assertEqual(mesh.periodic_bc, ("", 1))

# start tests
if __name__ == '__main__':
  unittest.main()
