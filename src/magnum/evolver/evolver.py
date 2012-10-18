class Evolver(object):
  def __init__(self, mesh):
    self.mesh = mesh
 
  def evolve(self, state, t_max):
    raise NotImplementedError("Evolver.evolve")
