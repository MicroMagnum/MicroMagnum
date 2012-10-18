from magnum.meshes import VectorField

import copy

class State(object):
  class Cache(object):
    pass

  def __init__(self, mesh):
    self.t = 0
    self.h = 0
    self.step = 0
    self.substep = 0
    self.mesh = mesh
    self.y = VectorField(mesh)
    self.flush_cache()
 
  def differentiate(self, dst):
    raise NotImplementedError("State.differentiate")

  cache = property(lambda self: self.__cache)

  def flush_cache(self):
    self.__cache = State.Cache()

  def finish_step(self):
    pass

  def clone(self, y_replacement = None):
    state = copy.copy(self)
    if y_replacement:
      state.y = y_replacement
    else:
      state.y = VectorField(self.mesh)
    state.flush_cache()
    return state
