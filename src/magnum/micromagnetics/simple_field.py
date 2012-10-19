import magnum.module as module
from magnum.mesh import VectorField
from magnum.logger import logger

class SimpleVectorField(module.Module):
  def __init__(self, var_id):
    super(SimpleVectorField, self).__init__()
    self.__var_id = var_id

  def params(self):
    return [self.__var_id]

  def initialize(self, system):
    logger.info("%s: Providing parameters %s" % (self.name(), ", ".join(self.params())))

    A = VectorField(system.mesh)
    A.clear()

    setattr(self, self.__var_id, A)

# This module is for use as an external field term in the LLG equation
class SimpleExternalField(SimpleVectorField):
  def __init__(self, var_id):
    super(SimpleExternalField, self).__init__(var_id)
    self.__var_id = var_id

  def properties(self):
    return {'EFFECTIVE_FIELD_TERM': self.__var_id}
