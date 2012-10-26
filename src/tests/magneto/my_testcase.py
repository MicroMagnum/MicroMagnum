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

from __future__ import print_function

import unittest

class MyTestCase(unittest.TestCase):
  def assertVectorFieldEqual(self, H_ref, H, epsilon = 1e-4):
    for i in range(H.size()):
      v1, v2 = H_ref.get(i), H.get(i)
      dx, dy, dz = abs(v1[0]-v2[0]), abs(v1[1]-v2[1]), abs(v1[2]-v2[2])
      if dx >= epsilon or dy >= epsilon or dz >= epsilon: print("\n", i, dx, dy, dz, "\n")
      self.assertTrue(dx < epsilon and dy < epsilon and dz < epsilon)
