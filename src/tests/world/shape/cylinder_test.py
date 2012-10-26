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

import math

class CylinderTest(unittest.TestCase):
   def setUp(self):
     P1, P2, R = (0,0,0), (40,0,0), 10
     self.cyl1 = Cylinder(P1, P2, R)

   def test_isPointInside(self):
     f = 10.0 / math.sqrt(2.0)
     for P in [(10,0,0), (20,0,0), (30,0,0), (10,f-0.1,f-0.1), (10,-f+0.1,-f+0.1)]:
       self.assertTrue(self.cyl1.isPointInside(P))
     for P in [(-10,0,0), (50,0,0), (10,f+0.1,f+0.1), (10,-f-0.1,-f-0.1)]:
       self.assertFalse(self.cyl1.isPointInside(P))

# start tests
if __name__ == '__main__':
  unittest.main()
