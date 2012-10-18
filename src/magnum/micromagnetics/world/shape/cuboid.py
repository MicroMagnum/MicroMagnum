from .shape import Shape

class Cuboid(Shape):
  def __init__(self, p1, p2):
    super(Cuboid, self).__init__()

    # Reorder coordinates in points
    def sort2(tup):
      a, b = tup
      if a < b:
        return a,b
      else:
        return b,a
    self.__p1, self.__p2 = zip(*map(sort2, zip(p1, p2)))

  def getBoundingBox(self):
    return self.__pos1, self.__pos2

  def isPointInside(self, pt):
    p1, p2 = self.__p1, self.__p2
    return pt[0] >= p1[0] and pt[0] < p2[0] and pt[1] >= p1[1] and pt[1] < p2[1] and pt[2] >= p1[2] and pt[2] < p2[2]

  def __repr__(self):
    return "Cuboid(" + repr(self.__p1) + ", " + repr(self.__p2) + ")"
