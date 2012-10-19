#!/usr/bin/python
from magnum import *
import unittest

class ExternalFieldTest(unittest.TestCase):

  def setUp(self):
    world = World(RectangularMesh((10,10,10),(1e-9,1e-9,1e-9)))
    self.solver = create_solver(world, [ExternalField])

  def test_uniform_field(self):
    self.solver.state.H_ext_amp   = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_phase = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_freq  = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_offs  = (1.0, 2.0, 3.0)
    self.solver.state.H_ext_fn    = None
    self.assertEqual((1.0, 2.0, 3.0), self.solver.state.H_ext.average())

  def test_uniform_user_fn_field(self):
    self.solver.state.H_ext_amp   = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_phase = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_freq  = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_offs  = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_fn    = lambda t: (1.0, 2.0, 3.0)
    self.assertEqual((1.0, 2.0, 3.0), self.solver.state.H_ext.average())

  def test_nonuniform_user_fn_field(self):
    H_ext = VectorField(self.solver.world.mesh)
    H_ext.fill((1.0, 2.0, 3.0)) # this counts as non-uniform because H_ext is not a Python 3-vector.

    self.solver.state.H_ext_amp   = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_phase = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_freq  = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_offs  = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_fn    = lambda t: H_ext
    self.assertEqual((1.0, 2.0, 3.0), self.solver.state.H_ext.average())

  def test_incompatible_parameters(self):
    self.solver.state.H_ext_amp   = (1.0, 0.0, 0.0)
    self.solver.state.H_ext_phase = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_freq  = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_offs  = (0.0, 0.0, 0.0)
    self.solver.state.H_ext_fn    = lambda t: (1,2,3)
    try: 
      # Error: Can't have H_ext_fn and other parameters non-zero at the same time.
      foo = self.solver.state.H_ext.average()
    except ValueError as e: # this is expected
      pass 
    else:
      self.fail()

if __name__ == '__main__':
  unittest.main()
