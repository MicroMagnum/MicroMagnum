from magnum.meshes import VectorField

from .evolver import Evolver

class Euler(Evolver):
  def __init__(self, mesh, step_size):
    super(Euler, self).__init__(mesh)
    self.step_size = float(step_size)

  def evolve(self, state, t_max):
    dydt = state.differentiate()
    state.y.add(dydt, self.step_size)
    state.t += self.step_size
    state.h = self.step_size
    state.step += 1
    state.substep = 0
    state.flush_cache()
    state.finish_step()
    return state
