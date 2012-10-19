import magnum.module as module
import magnum.magneto as magneto
from magnum.mesh import VectorField, Field

# void fdm_slonchewski(
# 	int dim_x, int dim_y, int dim_z,
# 	double delta_x, double delta_y, double delta_z,
# 	double a_j,
# 	const VectorMatrix &p, // spin polarization
# 	const Matrix &Ms,
# 	const Matrix &alpha,
# 	const VectorMatrix &M,
# 	VectorMatrix &dM
# );

class MacroSpinTorque(module.Module):
  def __init__(self, do_precess = True):
    super(SpinTorque, self).__init__()
    self.__do_precess = do_precess
      
  def calculates(self):
    return ["dMdt_ST"]

  def params(self):
    return ["a_j", "p"]

  def properties(self):
    return {'LLGE_TERM': "dMdt_ST"}

  def initialize(self, system):
    self.system = system
    self.a_j = Field(self.system.mesh); self.a_j.fill(0.0)
    self.p = Field(self.system.mesh); self.p.fill(0.0)

  def calculate(self, state, id):
    cache = state.cache

    if id == "dMdt_ST":
      if hasattr(cache, "dMdt_ST"): return cache.dMdt_ST
      dMdt_ST = cache.dMdt_ST = VectorField(self.system.mesh)

      # Calculate macro spin torque term due to Slonchewski
      nx, ny, nz = self.system.mesh.num_nodes
      dx, dy, dz = self.system.mesh.delta
      magneto.fdm_slonchewski(
        nx, ny, nz, dx, dy, dz, #self.__do_precess,
        a_j, p, state.Ms, state.alpha,
        state.M, dMdt_ST
      )
      return dMdt_ST

    else:
      raise KeyError("MacroSpinTorque.calculate: Can't calculate %s", id)
