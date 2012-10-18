#!/usr/bin/python
from magnum import *
import unittest

# World
class WorldTest(unittest.TestCase):
  def setUp(self):
    self.mesh = RectangularMesh((100, 100, 1), (1e-9, 1e-9, 1e-9))
    self.world = World(
      self.mesh,
      Body("body1", Material.Py(), Everywhere()),
      Body("body2", Material.Py(), Cylinder((0,0,0), (0,50e-9,0), 20e-9))
    )

  def test_findBody(self):
    body1 = self.world.findBody("body1")
    self.assertTrue(isinstance(body1, Body))
    self.assertEqual(body1.id, "body1")
    try:
      body3 = self.world.findBody("body3")
      self.fail()
    except: pass

  def test_getters(self):
    self.assertEqual(len(self.world.bodies), 2)
    self.assertTrue(isinstance(self.world.mesh, RectangularMesh))

  def test_construction(self):
    # Ok.
    world = World(self.mesh, Material.Py())
    self.assertEqual(len(world.bodies), 1)

    # Not ok: Need direct Material if no body is given.
    try:
      world = World(self.mesh) 
      self.fail()
    except: pass

    # Not ok: Duplicate Material
    try:
      world = World(self.mesh, Material.Py(), Material.Py())
      self.fail()
    except: pass

    # Not ok: Material and Body(s) given
    try:
      world = World(self.mesh, Material.Py(), Body("body1", Material.Py()))
      self.fail()
    except: pass

# start tests
if __name__ == '__main__':
  unittest.main()
