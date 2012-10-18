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
