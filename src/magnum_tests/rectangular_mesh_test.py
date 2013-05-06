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

from magnum import RectangularMesh
import unittest

class RectangularMeshTest(unittest.TestCase):

    def test_properties(self):
        mesh = RectangularMesh((1,2,3),(4.0,5.0,6.0))
        self.assertEqual(mesh.num_nodes, (1,2,3))
        self.assertEqual(mesh.delta, (4.0,5.0,6.0))

    def test_derived_quantities(self):
        mesh = RectangularMesh((10,10,10),(2,2,2))
        self.assertEqual(mesh.cell_volume, 8)
        self.assertEqual(mesh.volume, 8000)
        self.assertEqual(mesh.size, (20,20,20))
        self.assertEqual(mesh.delta, (2,2,2))
        self.assertEqual(mesh.num_nodes, (10,10,10))
        self.assertEqual(mesh.total_nodes, 1000)
        self.assertEqual(mesh.periodic_bc, ("", 1))

if __name__ == '__main__':
    unittest.main()
