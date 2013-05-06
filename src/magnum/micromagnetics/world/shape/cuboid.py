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

from .shape import Shape

class Cuboid(Shape):
    """
    A cuboid shape. It is defined by the coordinates of two
    (arbitrary) diagonally opposite cuboid vertices. The cuboid is always
    orthogonal to the coordinate axes.
    """

    def __init__(self, p1, p2):
        super(Cuboid, self).__init__()

        # Reorder coordinates in points
        def sort2(tup):
            a, b = tup
            if a < b:
                return a,b
            else:
                return b,a
        self.__p1, self.__p2 = zip(*map(sort2, zip(p1, p2)))

    def getBoundingBox(self):
        return self.__p1, self.__p2

    def isPointInside(self, pt):
        p1, p2 = self.__p1, self.__p2
        return pt[0] >= p1[0] and pt[0] < p2[0] and pt[1] >= p1[1] and pt[1] < p2[1] and pt[2] >= p1[2] and pt[2] < p2[2]

    def __repr__(self):
        return "Cuboid(" + repr(self.__p1) + ", " + repr(self.__p2) + ")"
