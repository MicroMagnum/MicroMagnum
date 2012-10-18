#!/usr/bin/python
from magnum import *

world = World(
  RectangularMesh((125, 50, 1), (4e-9, 4e-9, 20e-9)),
  Body("vortex1", Material.Py(P=0.1, xi=0.02), Cuboid((  0e-9, 0e-9, 0e-9), (200e-9, 200e-9, 20e-9))),
  Body("vortex2", Material.Py(P=0.1, xi=0.02), Cuboid((300e-9, 0e-9, 0e-9), (500e-9, 200e-9, 20e-9)))
)

def run(polarization1, polarization2, log_file):
  solver = create_solver(world, [StrayField, ExchangeField], log=True)
  solver.state["vortex1"].M = vortex.magnetizationFunction(100e-9, 100e-9, polarization1)
  solver.state["vortex2"].M = vortex.magnetizationFunction(400e-9, 100e-9, polarization2)
  solver.state.alpha = 0.5
  solver.relax()
  M = solver.state.M
  
  solver = create_solver(world, [StrayField, ExchangeField, SpinTorque, AlternatingCurrent], log=True)
  #solver = create_solver(world, [StrayField, ExchangeField], log=True)
  solver.state.M = M
  solver.state.j_amp  = (1e11, 0, 0)
  solver.state.j_freq = (4.4e9, 0, 0)

  log = DataTableLog(log_file)
  log.addColumn(("V1x", "Vortex 1 core x-pos.", "m"), ("V1y", "Vortex 1 core y-pos.", "m"), lambda s: vortex.findCore(solver, 100e-9, 100e-9, "vortex1"))
  log.addColumn(("V2x", "Vortex 2 core x-pos.", "m"), ("V2y", "Vortex 2 core y-pos.", "m"), lambda s: vortex.findCore(solver, 400e-9, 100e-9, "vortex2"))
  solver.addStepHandler(log, condition.EveryNthStep(20)) # add logger to solver

  solver.solve(condition.Time(10e-9))

run(+1, +1, "coupled-vortices-same-pol.odt")
run(+1, -1, "coupled-vortices-diff-pol.odt")
