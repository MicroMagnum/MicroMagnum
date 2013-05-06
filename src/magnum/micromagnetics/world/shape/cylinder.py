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

class Cylinder(Shape):
    """
    A cylindric shape. The cylinder is defined by two points and a radius.
    """

    def __init__(self, p1, p2, r):
        super(Cylinder, self).__init__()

        # Make sure we are working with floating point numbers..
        p1, p2 = tuple(map(float, p1)), tuple(map(float, p2))

        # Initilaize determination of the perpendicular connection from an arbitrary point
        # to the line between p1 and p2
        u  = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        u2 = u[0]*u[0] + u[1]*u[1] + u[2]*u[2]
        h  = (p1[0]*u[0] + p1[1]*u[1] + p1[2]*u[2])/u2

        self.__p1 = p1
        self.__p2 = p2
        self.__r = r
        self.__u = u
        self.__u2 = u2
        self.__h = h

    def isPointInside(self, pt):
        p1, r, u, u2, h = self.__p1, self.__r, self.__u, self.__u2, self.__h
        d = (u[0]*pt[0] + u[1]*pt[1] + u[2]*pt[2]) / u2

        s = d - h
        if s >= 0 and s<=1:
            f = (p1[0] + s*u[0], p1[1] + s*u[1], p1[2] + s*u[2])
            f_pt = (f[0] - pt[0], f[1] - pt[1], f[2] - pt[2])
            dis = f_pt[0]*f_pt[0] + f_pt[1]*f_pt[1] + f_pt[2]*f_pt[2]
            return dis <= r**2
        else:
            return False

    def __repr__(self):
        return "Cylinder(" + repr(self.__p1) + ", " + repr(self.__p2) + ", " + repr(self.__r) + ")"
