from __future__ import print_function

import unittest

class MyTestCase(unittest.TestCase):
  def assertVectorFieldEqual(self, H_ref, H, epsilon = 1e-4):
    for i in range(H.size()):
      v1, v2 = H_ref.get(i), H.get(i)
      dx, dy, dz = abs(v1[0]-v2[0]), abs(v1[1]-v2[1]), abs(v1[2]-v2[2])
      if dx >= epsilon or dy >= epsilon or dz >= epsilon: print("\n", i, dx, dy, dz, "\n")
      self.assertTrue(dx < epsilon and dy < epsilon and dz < epsilon)
