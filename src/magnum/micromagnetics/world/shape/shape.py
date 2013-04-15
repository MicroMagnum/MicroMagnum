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

####################################################################
# Shape classes                                                    #
####################################################################

# Classes:
#
#   Shape [abstract]
#    |- Cuboid
#    |- Cylinder
#    |- Everywhere
#    |- ImageShape
#    |- Prism
#    |- UnionShape     (compound shape)
#    |- IntersectShape (compound shape)
#    |- InvertedShape  (compound shape)

class Shape(object):
    """
    The abstract Shape class defines a geometrical shape in 3-D space.
    Shapes are used to define regions in the simulated volume in order to set up spatially varying material parameters (see World and Body classes).
    The most important method is *isPointInside*, which has to be implemented by concrete subclasses.

    There are predefinded shapes to use and change.

    Complex shapes can be created from atomic shapes via set operations: union, intersect, invert.
    (see e.g. http://en.wikipedia.org/wiki/Constructive_solid_geometry )
    """

    def getCellIndices(self, mesh):
        # linear index list of all mesh nodes
        all_indices = range(0, mesh.total_nodes)
        # list of linear indexes of mesh nodes which are inside the shape
        inside_indices = [idx for idx in all_indices if self.isPointInside(mesh.getPosition(idx))]
        return inside_indices

    def isPointInside(self, pt):
        raise NotImplementedError("A shape class needs to implement isPointInside")

    def union(self, other):
        return UnionShape(self, other)

    def intersect(self, other):
        return IntersectShape(self, other)

    def invert(self):
        return InvertedShape(self)

    ### Operators to call union, intersect and invert ########

    def __or__ (self, other): return self.union(other)
    def __and__(self, other): return self.intersect(other)
    def __not__(self): return self.invert()

class InvertedShape(Shape):
    def __init__(self, a):
        super(InvertedShape, self).__init__()
        self.__a = a

    def isPointInside(self, pt):
        return not self.__a.isPointInside(pt)

    def __repr__(self):
        return "invert(" + repr(self.__a) + ")"

class UnionShape(Shape):
    def __init__(self, a, b):
        super(UnionShape, self).__init__()
        self.__a, self.__b = a, b

    def isPointInside(self, pt):
        return self.__a.isPointInside(pt) or self.__b.isPointInside(pt)

    def __repr__(self):
        return "union(" + repr(self.__a) + ", " + repr(self.__b) + ")"

class IntersectShape(Shape):
    def __init__(self, a, b):
        super(IntersectShape, self).__init__()
        self.__a, self.__b = a, b

    def isPointInside(self, pt):
        return self.__a.isPointInside(pt) and self.__b.isPointInside(pt)

    def __repr__(self):
        return "intersection(" + repr(self.__a) + ", " + repr(self.__b) + ")"
