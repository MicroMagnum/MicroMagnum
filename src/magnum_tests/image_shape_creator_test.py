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

from magnum import RectangularMesh, ImageShapeCreator
import unittest

try:
    import Image
except ImportError: # if python imaging lib is not installed
    pass
else:
    class ImageShapeTest(unittest.TestCase):
        def setUp(self):
            mesh = RectangularMesh((100, 100, 1), (1, 1, 1))
            isc = ImageShapeCreator("image_shape_creator_test.png", mesh)

            self.shape0 = isc.pick("blue")  # bottom-left of img -> (0,0)
            self.shape1 = isc.pick("green") # bottom-right -> (99,0)
            self.shape2 = isc.pick("black") # top-left -> (0,99)
            self.shape3 = isc.pick("red")   # top-right -> (99,99)

        def test_2d(self):
            mesh = RectangularMesh((100, 100, 1), (1, 1, 1))
            self.assertEquals([ 0+ 0*100], self.shape0.getCellIndices(mesh))
            self.assertEquals([99+ 0*100], self.shape1.getCellIndices(mesh))
            self.assertEquals([ 0+99*100], self.shape2.getCellIndices(mesh))
            self.assertEquals([99+99*100], self.shape3.getCellIndices(mesh))

if __name__ == '__main__':
    unittest.main()
