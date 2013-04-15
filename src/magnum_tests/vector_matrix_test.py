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

from magnum import *
from magnum.magneto import VectorMatrix, Shape

import unittest

class VectorMatrixTest(unittest.TestCase):

    def test_fill(self):
        m1 = VectorMatrix(Shape(10, 10, 10));
        m1.fill((1.0, 2.0, 3.0));

        self.assertEqual(m1.get(4, 4, 4), (1.0, 2.0, 3.0))
        self.assertEqual(m1.uniform_value, (1.0, 2.0, 3.0))
        self.assertTrue(m1.isUniform())

        m1.set(4, 4, 4, (3.0, 2.0, 1.0))
        self.assertEqual(m1.get(4, 4, 4), (3.0, 2.0, 1.0))
        self.assertEqual(m1.get(0, 1, 2), (1.0, 2.0, 3.0))
        self.assertFalse(m1.isUniform())

    def test_clear(self):
        A = VectorMatrix(Shape(100,100,10))
        A.fill((1.0, 2.0, 3.0))
        A.clear()
        for idx in range(A.size()):
            self.assertTrue(A.get(idx) == (0.0, 0.0, 0.0))

    def test_fill(self):
        A = VectorMatrix(Shape(100,100,10))
        A.fill((1.0, 2.0, 3.0))
        for idx in range(A.size()):
            self.assertTrue(A.get(idx) == (1.0, 2.0, 3.0))

    def test_add_0(self):
        A = VectorMatrix(Shape(100,100,10))
        B = VectorMatrix(Shape(100,100,10))
        for idx in range(A.size()):
            A.set(idx, (1*idx, 2*idx, 3*idx))
            B.set(idx, (4*idx, 5*idx, 6*idx))
        A.add(B)
        for idx in range(A.size()):
            self.assertEqual(A.get(idx), (5*idx, 7*idx, 9*idx))

    def test_add_1(self):
        A = VectorMatrix(Shape(100,100,10))
        B = VectorMatrix(Shape(100,100,10))
        for idx in range(A.size()):
            A.set(idx, (1*idx, 2*idx, 3*idx))
            B.set(idx, (4*idx, 5*idx, 6*idx))
        A.add(B, 2.0)
        for idx in range(A.size()):
            self.assertEqual(A.get(idx), (9*idx, 12*idx, 15*idx))

    def test_dotSum(self):
        A = VectorMatrix(Shape(10,10,10))
        B = VectorMatrix(Shape(10,10,10))

        for idx in range(A.size()):
            A.set(idx, (0.1*idx, 0.2*idx, -0.3*idx))
            B.set(idx, (-0.4*idx, 0.5*idx, -0.6*idx))
        self.assertTrue(abs(A.dotSum(B) - 79880040.0) < 0.1)

    def test_absMax(self):
        A = VectorMatrix(Shape(10,10,10))
        for idx in range(A.size()):
            A.set(idx, (10,10,10))
        A.set(500, (10,20,30))
        self.assertTrue(A.absMax() - 37.41657387 < 0.0001)

    def test_average(self):
        A = VectorMatrix(Shape(10,10,10))
        for idx in range(A.size()):
            A.set(idx, (idx, 2*idx, 3*idx))
        self.assertTrue(A.average() == (999.0/2,2*999.0/2,3*999.0/2))

if __name__ == '__main__':
    unittest.main()
