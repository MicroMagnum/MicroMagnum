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

from .material import Material
from .shape import Shape, Everywhere

class Body(object):
  """
  A Body consists of an Id, a Shape and a Material.
  """

  def __init__(self, body_id, material, shape = None):
    assert isinstance(body_id, str)
    assert isinstance(material, Material)
    assert isinstance(shape, (Shape, type(None)))

    self.__id       = body_id
    self.__material = material
    self.__shape    = shape or Everywhere()

  @property
  def material(self): 
    """
    Get the material of this body.

    .. code-block:: python 

       print(body.material)
    """
    return self.__material

  @property
  def shape(self): 
    """
    Get the shape of this body.
    """
    return self.__shape

  @property
  def id(self): 
    """
    Get the Id of this body.
    """
    return self.__id

  def __repr__(self):
    return "Body(" + repr(self.__id) + ", " + repr(self.__material) + ", " + repr(self.__shape) + ")"
