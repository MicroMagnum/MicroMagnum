#!/usr/bin/python 
from magnum import *
from math import pi, cos, sin

############################################################
# Create a world (choose one of the discretizations)       #
############################################################

# 500 x 125 x 3 nm^3
#world = World(RectangularMesh((512, 512, 1), (500.0/256*1e-9, 125.0/256*1e-9, 3e-9)), Body("all", Material.Py(alpha=0.02)))
#world = World(RectangularMesh((200, 100, 1), (2.5e-9, 1.25e-9, 3.0e-9)), Body("all", Material.Py(alpha=0.02)))
#world = World(RectangularMesh((200,  50, 1), (2.5e-9, 2.50e-9, 3.0e-9)), Body("all", Material.Py(alpha=0.02)))
world = World(RectangularMesh((100,  25, 1), (  5e-9,    5e-9, 3.0e-9)), Body("all", Material.Py(alpha=0.02)))

############################################################
# Relax an s-state as the initial magnetization of the SP4 #
############################################################

def make_initial_sp4_state():
  # Specify an s-state-like starting state
  def state0(field, pos): 
    u = abs(pi*(pos[0]/field.mesh.size[0]-0.5)) / 2.0
    return 8e5 * cos(u), 8e5 * sin(u), 0
  # Relax to get initial state for SP4
  solver = create_solver(world, [StrayField, ExchangeField], log=True, do_precess=False, evolver="rkf45", eps_abs=1e-4, eps_rel=1e-2)
  solver.state.M = state0
  solver.state.alpha = 0.5
  solver.relax(1.0)
  return solver.state.M # return the final magnetization

############################################################
# Apply an external field H on initial magnetization M0    #
############################################################

def apply_field(M0, H, file_prefix):

  class ZeroCrossChecker(StepHandler):
    def __init__(self):
      self.crossed = False
    def handle(self, state):
      Mx = state.M.average()[0]
      if not self.crossed and Mx < 0.0:
        t_0, Mx_0 = self.t_0, self.Mx_0
        t_1, Mx_1 = state.t, Mx
        print ("First zero-crossing of <Mx> at", (t_0 + Mx_0*(t_1-t_0)/(Mx_1-Mx_0))*1e9, "ns!")
        writeOMF(file_prefix + "-Mx-zero.omf", state.M)
        writeImage(file_prefix + "-Mx-zero.png", state.M, "x")
        self.crossed = True
      else:
        self.t_0, self.Mx_0 = state.t, Mx

  #solver = create_solver(world, [StrayField, ExchangeField, ExternalField], log=True, evolver="rk23", eps_abs=1e-2, eps_rel=1e-2)
  #solver = create_solver(world, [StrayField, ExchangeField, ExternalField], log=True, eps_abs=1e-2, eps_rel=1e-2)
  solver = create_solver(world, [StrayField, ExchangeField, ExternalField], log=True, evolver="rkf45", eps_abs=1e-4, eps_rel=1e-4)
  solver.state.M = M0
  solver.state.H_ext_offs = H
  solver.addStepHandler(ZeroCrossChecker(), condition.Always())
  dtsh = DataTableLog(file_prefix + ".odt", title = file_prefix)
  dtsh.addEnergyColumn("E_stray")
  dtsh.addEnergyColumn("E_exch")
  dtsh.addEnergyColumn("E_ext")
  dtsh.addEnergyColumn("E_tot")
  solver.addStepHandler(dtsh, condition.EveryNthStep(10))
  solver.solve(condition.Time(1.0e-9))

############################################################
# Main program                                             #
############################################################

M0 = make_initial_sp4_state()
writeOMF("sp4_M0.omf", M0)
#M0 = readOMF("sp4_M0.omf")
apply_field(M0, (-24.6e-3/MU0, +4.3e-3/MU0, 0.0), "sp4-1")
apply_field(M0, (-35.5e-3/MU0, -6.3e-3/MU0, 0.0), "sp4-2")
