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

from magnum import Body, Material, Cuboid, Everywhere
import unittest

class BodyTest(unittest.TestCase):

    def test_construction(self):
        body1 = Body("body1", Material.Py(), Cuboid((0,0,0), (1,1,1)))
        self.assertEqual(body1.id, "body1")
        self.assertTrue(isinstance(body1.material, Material))
        self.assertTrue(isinstance(body1.shape, Cuboid))

    def test_default_shape_is_everywhere(self):
        body2 = Body("body2", Material.Py())
        self.assertTrue(isinstance(body2.shape, Everywhere))

if __name__ == '__main__':
    unittest.main()
