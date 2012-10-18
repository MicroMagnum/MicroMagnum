import magnum.module as module
import magnum.magneto as magneto
from magnum.meshes import VectorField, Field
from .constants import MU0

class AnisotropyField(module.Module):
  def __init__(self):
    super(AnisotropyField, self).__init__()

  def calculates(self):
    return ["H_aniso", "E_aniso"]

  def params(self):
    return ["k_uniaxial", "k_cubic", "axis1", "axis2"]

  def properties(self):
    return {'EFFECTIVE_FIELD_TERM': "H_aniso", 'EFFECTIVE_FIELD_ENERGY': "E_aniso"}

  def initialize(self, system):
    self.system = system
    self.k_uniaxial = Field(self.system.mesh); self.k_uniaxial.fill(0.0)
    self.k_cubic = Field(self.system.mesh); self.k_cubic.fill(0.0)
    self.axis1 = VectorField(self.system.mesh); self.axis1.fill((0.0, 0.0, 0.0))
    self.axis2 = VectorField(self.system.mesh); self.axis2.fill((0.0, 0.0, 0.0))

  def calculate(self, state, id):
    if id == "H_aniso":
      if hasattr(state.cache, "H_aniso"): return state.cache.H_aniso
      H_aniso = state.cache.H_aniso = VectorField(self.system.mesh)
      
      axis1 = self.axis1
      axis2 = self.axis2
      k_uni = self.k_uniaxial
      k_cub = self.k_cubic
      
      skip_uni = k_uni.isUniform() and k_uni.uniform_value == 0.0
      have_uni = not skip_uni
      skip_cub = k_cub.isUniform() and k_cub.uniform_value == 0.0
      have_cub = not skip_cub
  
      Ms = self.system.Ms    
  
      if   not have_uni and not have_cub:
        H_aniso.fill((0.0, 0.0, 0.0))
      elif not have_uni and have_cub:
        state.cache.E_aniso_sum = magneto.cubic_anisotropy(axis1, axis2, k_cub, Ms, state.M, H_aniso)
      elif have_uni and not have_cub:
        state.cache.E_aniso_sum = magneto.uniaxial_anisotropy(axis1, k_uni, Ms, state.M, H_aniso)
      elif have_uni and have_cub:
        tmp = VectorField(self.system.mesh)
        E0 = magneto.uniaxial_anisotropy(axis1, k_uni, Ms, state.M, tmp)
        E1 = magneto.cubic_anisotropy(axis1, axis2, k_cub, Ms, state.M, H_aniso)
        state.cache.E_aniso_sum = E0 + E1
        H_aniso.add(tmp)

      return H_aniso
  
    elif id == "E_aniso":
      if not hasattr(state.cache, "E_aniso"):
        foo = state.H_aniso
      return state.cache.E_aniso_sum * self.system.mesh.cell_volume

    else:
      raise KeyError("AnisotropyField.calculate: Can't calculate %s", id)
