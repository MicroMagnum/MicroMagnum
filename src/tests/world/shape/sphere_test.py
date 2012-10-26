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

class SphereTest(unittest.TestCase):
  def setUp(self):
    self.sphere = Sphere((1,1,1), 1)
    self.mesh = RectangularMesh((20, 20, 20), (0.1, 0.1, 0.1))

  def test_isPointInside(self):
    self.assertTrue (self.sphere.isPointInside((1.0, 1.0, 1.0)))
    self.assertTrue (self.sphere.isPointInside((1.9, 1.0, 1.0)))
    self.assertTrue (self.sphere.isPointInside((1.0, 0.1, 1.0)))
    self.assertTrue (self.sphere.isPointInside((1.0, 1.0, 1.9)))
    self.assertFalse(self.sphere.isPointInside((2.1, 1.0, 1.0)))
    self.assertFalse(self.sphere.isPointInside((1.9, 1.9, 1.0)))
    self.assertFalse(self.sphere.isPointInside((0.1, 0.1, 1.0)))

# start tests
if __name__ == '__main__':
  unittest.main()
