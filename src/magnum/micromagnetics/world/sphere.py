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

from magnum.micromagnetics.world.shape import Shape

class Sphere(Shape):
    """
    A sphere defined by its midpoint position and its radius.
    """

    def __init__(self, position, radius):
        """
        Constructor: Sphere(position, radius).

        E.g.: Sphere((2e-9, 4e-9, 7e-9), 1e-9))
        """
        super(Shape, self).__init__()

        self.x, self.y, self.z = position
        self.r = radius

    def getBoundingBox(self):
        x, y, z, r = self.x, self.y, self.z, self.r
        return ((x - r, y - r, z - r), (x + r, y + r, z + r))

    def isPointInside(self, pt):
        dx = self.x - pt[0]
        dy = self.y - pt[1]
        dz = self.z - pt[2]
        return dx**2 + dy**2 + dz**2 < self.r**2

    def __repr__(self):
        return "Sphere(%s, %s)" % ((self.x, self.y, self.z), self.r)
