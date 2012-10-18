#!/usr/bin/python
from magnum import *
import unittest

# Body
class BodyTest(unittest.TestCase):
  def setUp(self):
    pass

  def test_construction(self):
    body1 = Body("body1", Material.Py(), Cuboid((0,0,0), (1,1,1)))
    self.assertEqual(body1.id, "body1")
    self.assertTrue(isinstance(body1.material, Material))
    self.assertTrue(isinstance(body1.shape, Cuboid))

    body2 = Body("body2", Material.Py())
    self.assertTrue(isinstance(body2.shape, Everywhere))

# start tests
if __name__ == '__main__':
  unittest.main()
