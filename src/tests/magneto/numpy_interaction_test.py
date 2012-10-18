#!/usr/bin/python
from magnum import *
from magnum.magneto import Matrix, VectorMatrix, Shape

import unittest
import itertools

try:
  import numpy as np
except ImportError:
  class NumpyInteractionTest(unittest.TestCase):
    pass
else:
  class NumpyInteractionTest(unittest.TestCase):
 
    def test_matrix_to_numpy_1(self):
      A = Matrix(Shape(10,10,10))
      A.fill(42.0)

      B = A.to_numpy()
      self.assertEquals((10,10,10), B.shape)
      self.assertTrue(np.all(B[:,:,:] == 42.0))

    def test_matrix_to_numpy_2(self):
      def flub(x,y,z): return x*y*z + 42 + (5*x)*(2-y) + 8*z
  
      A = Matrix(Shape(10,20,30))
      for x,y,z in itertools.product(range(10), range(20), range(30)):
        A.set(x,y,z, flub(x,y,z))
      
      B = A.to_numpy()
      self.assertEquals(A.shape, B.shape)
      for x,y,z in itertools.product(range(10), range(20), range(30)):
        self.assertEquals(flub(x,y,z), B[x,y,z])

    def test_vectormatrix_to_numpy_1(self):
      A = VectorMatrix(Shape(10,10,10))
      A.fill((1.0, 2.0, 3.0))
      
      B = A.to_numpy()
      self.assertEquals((10,10,10,3), B.shape)
      self.assertTrue(np.all(B[:,:,:,0] == 1.0))
      self.assertTrue(np.all(B[:,:,:,1] == 2.0))
      self.assertTrue(np.all(B[:,:,:,2] == 3.0))
  
    def test_vectormatrix_to_numpy_2(self):
      def flub(x,y,z): return x*y*z, 42, (5*x)*(2-y) + 8*z
  
      A = VectorMatrix(Shape(10,20,30))
      for x,y,z in itertools.product(range(10), range(20), range(30)):
        A.set(x,y,z, flub(x,y,z))
      
      B = A.to_numpy()
      self.assertEquals(A.shape + (3,), B.shape)
      for x,y,z in itertools.product(range(10), range(20), range(30)):
        f = flub(x,y,z)
        self.assertEquals(f[0], B[x,y,z,0])
        self.assertEquals(f[1], B[x,y,z,1])
        self.assertEquals(f[2], B[x,y,z,2])

if __name__ == '__main__':
  unittest.main()
