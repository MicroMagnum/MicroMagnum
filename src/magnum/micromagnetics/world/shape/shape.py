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
#    |- OrShape        (compound shape)
#    |- AndShape       (compound shape)
#    |- InvertedShape  (compound shape)

class Shape(object):

  def getCellIndices(self, mesh):
    # linear index list of all mesh nodes
    all_indices = range(0, mesh.total_nodes)
    # list of linear indexes of mesh nodes which are inside the shape
    inside_indices = [idx for idx in all_indices if self.isPointInside(mesh.getPosition(idx))]
    return inside_indices

  def isPointInside(self, pt):
    raise NotImplementedError("A shape class needs to implement isPointInside")

  def combineOr(self, other):
    return OrShape(self, other)

  def combineAnd(self, other):
    return AndShape(self, other)

  def invert(self):
    return InvertedShape(self)

  ### Operators to call combineOr, combineAnd and invert ########

  def __or__ (self, other): return self.combineOr(other)
  def __and__(self, other): return self.combineAnd(other)  
  def __not__(self): return self.invert()

class InvertedShape(Shape):
  def __init__(self, a):
    Shape.__init__(self)
    self.__a = a

  def isPointInside(self, pt):
    return not self.__a.isPointInside(pt)

  def __repr__(self):
    return "InvertedShape(" + repr(self.__a) + ")"

class OrShape(Shape):
  def __init__(self, a, b):
    Shape.__init__(self)
    self.__a, self.__b = a, b

  def isPointInside(self, pt):
    return self.__a.isPointInside(pt) or self.__b.isPointInside(pt)

  def __repr__(self):
    return "OrShape(" + repr(self.__a) + ", " + repr(self.__b) + ")"

class AndShape(Shape):
  def __init__(self, a, b):
    Shape.__init__(self)
    self.__a, self.__b = a, b

  def isPointInside(self, pt):
    return self.__a.isPointInside(pt) and self.__b.isPointInside(pt)

  def __repr__(self):
    return "AndShape(" + repr(self.__a) + ", " + repr(self.__b) + ")"
