import magnum.module as module
import magnum.magneto as magneto

from magnum.logger import logger
from magnum.meshes import VectorField, Field

from .constants import GYROMAGNETIC_RATIO

import math

class LandauLifshitzGilbert(module.Module):
  def __init__(self, do_precess=True):
    super(LandauLifshitzGilbert, self).__init__()
    self.__do_precess = do_precess

  def calculates(self):
                                                           # deprecated names for H_tot and E_tot:
    return ["dMdt", "M", "H_tot", "E_tot", "deg_per_ns"] + ["H_eff", "E_eff"]

  def updates(self):
    return ["M"]

  def params(self):
    return ["Ms", "alpha"]

  def on_param_update(self, id):
    if id in ["Ms", "alpha"]:
      self.__valid_factors = False

  def initialize(self, system):
    self.system = system
    self.Ms = Field(system.mesh); self.Ms.fill(0.0)
    self.alpha = Field(system.mesh); self.alpha.fill(0.0)
    self.__valid_factors = False

    # Find other active modules
    self.field_terms = []
    self.field_energies = []
    self.llge_terms = []
    for mod in system.modules:
      prop = mod.properties()
      if 'EFFECTIVE_FIELD_TERM'   in prop.keys(): self.field_terms.append(prop['EFFECTIVE_FIELD_TERM'])
      if 'EFFECTIVE_FIELD_ENERGY' in prop.keys(): self.field_energies.append(prop['EFFECTIVE_FIELD_ENERGY'])
      if 'LLGE_TERM'              in prop.keys(): self.llge_terms.append(prop['LLGE_TERM'])

    logger.info("LandauLifshitzGilbert module configuration:")
    logger.info(" - H_tot = %s", " + ".join(self.field_terms) or "0")
    logger.info(" - E_tot = %s", " + ".join(self.field_energies) or "0")
    logger.info(" - dM/dt = %s", " + ".join(["LLGE(M, H_tot)"] + self.llge_terms) or "0")
    if not self.__do_precess: logger.info(" - Precession term is disabled")

  def calculate(self, state, id):
    if id == "M":
      return state.y
    if id == "H_tot" or id == "H_eff":
      return self.calculate_H_tot(state)
    if id == "E_tot" or id == "E_eff":
      return self.calculate_E_tot(state)
    if id == "dMdt":
      return self.calculate_dMdt(state)
    if id == "deg_per_ns":
      deg_per_timestep = (180.0 / math.pi) * math.atan2(state.dMdt.absMax() * state.h, state.M.absMax()) # we assume a<b at atan(a/b).
      deg_per_ns = 1e-9 * deg_per_timestep / state.h
      return deg_per_ns
    else:
      raise KeyError(id)

  def update(self, state, id, value):
    if id == "M":
      logger.info("Assigning new magnetic state M")
      module.assign(state.y, value)
      state.y.normalize(self.system.Ms)   # XXX: Is this a good idea? (Solution: Use unit magnetization everywhere.)
      state.flush_cache()
    else:
      raise KeyError(id)

  def calculate_H_tot(self, state):
    if hasattr(state.cache, "H_tot"): return state.cache.H_tot
    H_tot = state.cache.H_tot = VectorField(self.system.mesh)

    if len(self.field_terms) == 0:
      H_tot.fill((0.0, 0.0, 0.0))
    else: 
      for i, H_str in enumerate(self.field_terms):
        H_i = getattr(state, H_str)
        if i == 0:
          H_tot.assign(H_i)
        else: 
          H_tot.add(H_i)

    return H_tot

  def calculate_E_tot(self, state):
    if hasattr(state.cache, "E_tot"): return state.cache.E_tot
    state.cache.E_tot = sum(getattr(state, E_str) for E_str in self.field_energies) # calculate sum of all registered energy terms
    return state.cache.E_tot

  def calculate_dMdt(self, state):
    if not self.__valid_factors: self.__initFactors()

    if hasattr(state.cache, "dMdt"): return state.cache.dMdt
    dMdt = state.cache.dMdt = VectorField(self.system.mesh)

    # Get effective field
    H_tot = self.calculate_H_tot(state)
    
    # Basic term
    magneto.llge(self.__f1, self.__f2, state.M, H_tot, dMdt)
 
    # Optional other terms
    for dMdt_str in self.llge_terms:
      dMdt_i = getattr(state, dMdt_str)
      dMdt.add(dMdt_i)

    return dMdt

  def __initFactors(self):
    self.__f1 = f1 = Field(self.system.mesh) # precession factors
    self.__f2 = f2 = Field(self.system.mesh) # damping factors

    # Prepare factors    
    nx, ny, nz = self.system.mesh.num_nodes
    for z in range(nz):
      for y in range(ny):
        for x in range(nx):
          alpha, Ms = self.alpha.get(x,y,z), self.Ms.get(x,y,z)

          if Ms != 0.0:
            gamma_prime = GYROMAGNETIC_RATIO / (1.0 + alpha**2)
            f1_i = -gamma_prime
            f2_i = -alpha * gamma_prime / Ms
          else:
            f1_i, f2_i = 0.0, 0.0

          f1.set(x,y,z, f1_i)
          f2.set(x,y,z, f2_i)

    # If precession is disabled, blank f1.
    if not self.__do_precess: 
      f1.fill(0.0)

    # Done.
    self.__valid_factors = True
