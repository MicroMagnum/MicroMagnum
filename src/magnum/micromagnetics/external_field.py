from .alternating_field import AlternatingField
from .constants import MU0

class ExternalField(AlternatingField):
  def __init__(self):
    super(ExternalField, self).__init__("H_ext")

  def calculates(self):
    return super(ExternalField, self).calculates() + ["E_ext"]

  def properties(self):
    p = super(ExternalField, self).properties()
    p.update({'EFFECTIVE_FIELD_TERM': "H_ext", 'EFFECTIVE_FIELD_ENERGY': "E_ext"})
    return p

  def calculate(self, state, id):
    if id == "E_ext":
      return -MU0 * self.system.mesh.cell_volume * state.M.dotSum(state.H_ext)
    else:
      return super(ExternalField, self).calculate(state, id)
