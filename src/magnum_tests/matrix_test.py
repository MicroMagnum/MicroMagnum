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
from magnum.magneto import Matrix, Shape

import unittest
import itertools

class MatrixTest(unittest.TestCase):

    # TODO: methods to test:
    #   void clear();
    #   void assign(const Matrix &other);
    #   void scale(double factor);
    #   void add(const Matrix &op, double scale = 1.0);
    #   void multiply(const Matrix &rhs);
    #   void divide(const Matrix &rhs);
    #   void randomize();
    #   double maximum() const;
    #   double average() const;
    #   double sum() const;
    #   double getUniformValue() const;

    def test_fill(self):
        m1 = Matrix(Shape(10, 10, 10));

        m1.fill(1.0)
        self.assertEqual(m1.get(4, 4, 4), 1.0)
        self.assertEqual(m1.uniform_value, 1.0)
        self.assertTrue(m1.isUniform())

        m1.set(4, 4, 4, 2.0);
        self.assertEqual(m1.get(4, 4, 4), 2.0)
        self.assertFalse(m1.isUniform())

if __name__ == '__main__':
    unittest.main()
