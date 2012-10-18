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
