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

class Everywhere(Shape):
    """
    This shape describes the whole simulation volume, i.e. *isPointInside* always returns true.
    """

    def __init__(self):
        super(Everywhere, self).__init__()

    def getCellIndices(self, mesh):
        return range(0, mesh.total_nodes)

    def isPointInside(self, pt):
        return True

    def __repr__(self):
        return "Everywhere()"
