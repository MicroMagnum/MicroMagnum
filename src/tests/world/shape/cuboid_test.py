#!/usr/bin/python

# Copyright 2012 by the Micromagnum authors.
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

from magnum import *
import unittest

# Cuboid
class CuboidTest(unittest.TestCase):
  def setUp(self):
    self.cube = Cuboid((0,0,0), (1,1,1))
    self.mesh = RectangularMesh((10, 10, 10), (1.0, 1.0, 1.0))

  def test_isPointInside(self):
    self.assertTrue (self.cube.isPointInside(( 0.5,  0.5,  0.5)))
    self.assertFalse(self.cube.isPointInside((-0.5,  0.5,  0.5)))

# start tests
if __name__ == '__main__':
  unittest.main()
