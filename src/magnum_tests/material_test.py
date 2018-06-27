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

import unittest

from magnum import Material


class MaterialTest(unittest.TestCase):

    def test_getter(self):
        mat = Material({'Ms': 8e3})
        self.assertEqual(8e3, mat.Ms)

    def test_permalloy(self):
        py = Material.Py()
        self.assertEqual(8e5, py.Ms)

    def test_modified_permallow(self):
        mod_py = Material.Py(Ms=7e5)
        self.assertEqual(7e5, mod_py.Ms)

if __name__ == '__main__':
    unittest.main()
