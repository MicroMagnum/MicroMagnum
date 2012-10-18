#!/usr/bin/python
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
