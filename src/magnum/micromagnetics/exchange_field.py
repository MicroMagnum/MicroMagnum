import magnum.module as module
import magnum.magneto as magneto
from magnum.meshes import VectorField, Field
from .constants import MU0

class ExchangeField(module.Module):
  def __init__(self):
    super(ExchangeField, self).__init__()

  def calculates(self):
    return ["H_exch", "E_exch"]

  def params(self):
    return ["A"]

  def properties(self):
    return {'EFFECTIVE_FIELD_TERM': "H_exch", 'EFFECTIVE_FIELD_ENERGY': "E_exch"}

  def initialize(self, system):
    self.system = system
    self.A = Field(self.system.mesh); self.A.fill(0.0)
    self.__peri_x = system.mesh.periodic_bc[0].find("x") != -1
    self.__peri_y = system.mesh.periodic_bc[0].find("y") != -1
    self.__peri_z = system.mesh.periodic_bc[0].find("z") != -1

  def calculate(self, state, id):
    cache = state.cache

    if id == "H_exch":
      if hasattr(cache, "H_exch"): return cache.H_exch
      H_exch = cache.H_exch = VectorField(self.system.mesh)

      nx, ny, nz = self.system.mesh.num_nodes
      dx, dy, dz = self.system.mesh.delta
      bcx, bcy, bcz = self.__peri_x, self.__peri_y, self.__peri_z

      magneto.fdm_exchange(nx, ny, nz, dx, dy, dz, bcx, bcy, bcz, self.system.Ms, self.system.A, state.M, H_exch)
      return H_exch

    elif id == "E_exch":
      return -MU0/2.0 * self.system.mesh.cell_volume * state.M.dotSum(state.H_exch)

    else:
      raise KeyError("ExchangeField.calculate: Can't calculate %s", id)
