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

from magnum import create_controller
import unittest

class ControllerTest(unittest.TestCase):

    def test_empty_parameters(self):
        c = create_controller(lambda: None, [])
        self.assertEqual([], c.all_params)

    def test_simple_parameters(self):
        c = create_controller(lambda x: None, [1, 3, 5])
        self.assertEqual([(1,), (3,), (5,)], c.all_params)

    def test_product_parameters(self):
        c = create_controller(lambda x, y: None, [(1, [2, 3])])
        self.assertEqual([(1, 2), (1, 3)], c.all_params)

    def test_complicated_parameters(self):
        c = create_controller(lambda x, y: None, [(1, [2, 3]), (4, 5)])
        self.assertEqual([(1, 2), (1, 3), (4, 5)], c.all_params)

if __name__ == '__main__':
    unittest.main()
