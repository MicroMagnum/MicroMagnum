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

from magnum.micromagnetics.world.everywhere import Everywhere

class Body(object):
    """
    A Body object consists of a name (the id), a Shape and a Material. It
    specifies the material parameters of a region inside the simulation
    volume.
    """

    def __init__(self, id, material, shape=None):
        """
        Create a body object with an ID, a material, and a shape. If no
        shape is given, the Everywhere shape, which encompasses the whole
        simulation volume, is used as a default.
        """

        assert isinstance(id, str)

        self.__id       = id
        self.__material = material
        self.__shape    = shape or Everywhere()

    material = property(lambda self: self.__material)
    shape    = property(lambda self: self.__shape)
    id       = property(lambda self: self.__id)

    def __repr__(self):
        return "Body(" + repr(self.id) + ", " + repr(self.material) + ", " + repr(self.shape) + ")"
