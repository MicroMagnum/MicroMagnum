#!/usr/bin/python

# Copyright 2012, 2013 by the Micromagnum authors.
#
# This file is part of MicroMagnum.
#
# MicroMagnum is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MicroMagnum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.

from magnum import RectangularMesh, World, Body, Material, Everywhere, Cylinder
import unittest

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
            self.world.findBody("body3")
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
            World(self.mesh)
            self.fail()
        except:
            pass

        # Not ok: Duplicate Material
        try:
            world = World(self.mesh, Material.Py(), Material.Py())
            self.fail()
        except:
            pass

        # Not ok: Material and Body(s) given
        try:
            world = World(self.mesh, Material.Py(), Body("body1", Material.Py()))
            self.fail()
        except:
            pass

# start tests
if __name__ == '__main__':
    unittest.main()
