from .shape import *

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
