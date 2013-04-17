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

class Sphere(Shape):
    def __init__(self, p, r):
        super(Shape, self).__init__()
        self.__p = p
        self.__r = r

    def getBoundingBox(self):
        se = tuple(x - dx for x, dx in zip(self.__p, (self.__r, self.__r)))
        nw = tuple(x + dx for x, dx in zip(self.__p, (self.__r, self.__r)))
        return se, nw

    def isPointInside(self, pt):
        return sum((a - b)**2 for a, b in zip(pt, self.__p)) < self.__r**2

    def __repr__(self):
        return "Sphere(" + repr(self.__p) + ", " + repr(self.__r) + ")"
