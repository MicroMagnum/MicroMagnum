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
from .transformation import Transformation

from math import sqrt, acos

class Prism(Shape):
    def __init__(self, p1, p2, poly):
        super(Prism, self).__init__()
        self.T = Transformation()

        # Make sure we are working with floating point numbers..
        def make_float_tuple(x):
            return tuple(map(float, x))

        p1, p2 = make_float_tuple(p1), make_float_tuple(p2)
        poly = tuple(map(make_float_tuple, poly))

        # Determine new endpoint of the prism
        u  = (p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2])
        l_u = sqrt(u[0]**2 + u[1]**2 + u[2]**2)

        # Determine maximum distance of the prism to the center
        r = 0
        for p in poly:
            l_p2 = p[0]**2 + p[1]**2
            if l_p2 > r:
                r = l_p2
        r = sqrt(r)

        # Determine rotation angle and axis
        if l_u != 0:
            angle = acos(u[2]/l_u)
        else:
            angle = 0
        axis = (u[1],-u[0],0)
        l_axis = sqrt(axis[0]**2 + axis[1]**2)
        if l_axis != 0:
            norm_axis = (axis[0]/l_axis, axis[1]/l_axis, 0)
        else:
            norm_axis = (1, 0, 0)

        # Add translation and rotation matrix
        self.T.addTranslate(-p1[0], -p1[1], -p1[2])
        self.T.addRotate(angle, norm_axis)

        # Go on
        self.__p1 = p1
        self.__p2 = p2
        self.__poly = poly
        self.__u = u
        self.__l_u = l_u
        self.__r = r

    def isPointInside(self, pt):
        p1, p2, poly, u, l_u, r = self.__p1, self.__p2, self.__poly, self.__u, self.__l_u, self.__r

        # Transform investigated point
        pt = self.T.transformPoint(pt)

        # Deal with small numbers
        p = [pt[0], pt[1], pt[2]]
        for i in range(3):
            if abs(p[i]) < 1e-22:
                p[i] = 0
        pt = (p[0], p[1], p[2])

        # Deal with exclusions
        l_xy_pt = sqrt(pt[0]**2 + pt[1]**2)
        if pt[2] < 0 or pt[2] > l_u or l_xy_pt > r:
            return False

        # Test if point is inside Polygon
        x = pt[0]
        y = pt[1]

        n = len(poly)
        inside = False

        poly1x,poly1y = poly[0]
        for i in range(n+1):
            poly2x,poly2y = poly[i % n]
            if y > min(poly1y,poly2y):
                if y <= max(poly1y,poly2y):
                    if x <= max(poly1x,poly2x):
                        if poly1y != poly2y:
                            xinters = (y-poly1y)*(poly2x-poly1x)/(poly2y-poly1y)+poly1x
                        if poly1x == poly2x or x <= xinters:
                            inside = not inside
            poly1x,poly1y = poly2x,poly2y

        return inside

    def __repr__(self):
        return "Prism(" + repr(self.__p1) + ", " + repr(self.__p2) + ", " + repr(self.__poly) + ")"
