from magnum.meshes import VectorField
from magnum.module import Module

from .stray_field_calculator import DemagTensorField, StrayFieldCalculator
from .constants import MU0

class StrayField(Module):
  def __init__(self, method = "tensor"):
    super(StrayField, self).__init__()
    self.method = method
    self.padding = DemagTensorField.PADDING_ROUND_4

  def calculates(self):
    return ["H_stray", "E_stray"]

  def properties(self):
    return {'EFFECTIVE_FIELD_TERM': "H_stray", 'EFFECTIVE_FIELD_ENERGY': "E_stray"}

  def initialize(self, system):
    self.system = system
    self.calculator = StrayFieldCalculator(system.mesh, self.method, self.padding)

  def calculate(self, state, id):
    cache = state.cache

    if id == "H_stray":
      if hasattr(cache, "H_stray"): return cache.H_stray
      H_stray = cache.H_stray = VectorField(self.system.mesh)
      self.calculator.calculate(state.M, H_stray)
      return H_stray

    elif id == "E_stray":
      if hasattr(cache, "E_stray"): return cache.E_stray
      E_stray = cache.E_stray = -MU0/2.0 * self.system.mesh.cell_volume * state.M.dotSum(state.H_stray)
      return E_stray

    else:
      raise KeyError("StrayField.calculate: %s" % id)
