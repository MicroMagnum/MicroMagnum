#!/usr/bin/python
from magnum import *

world = World(
  RectangularMesh((50, 50, 1), (4e-9, 4e-9, 20e-9)),
  Body("all", Material.Py(), Everywhere())
)

# Relax
solver = create_solver(world, [StrayField, ExchangeField], log=True)
solver.state.M = vortex.magnetizationFunction(100e-9, 100e-9, 1)
solver.state.alpha = 0.3
solver.relax()
M = solver.state.M


# Excite the vortex
solver = create_solver(world, [StrayField, ExchangeField, ExternalField], log=True)
solver.state.M = M
solver.state.H_ext_amp = (10e-3/MU0, 0, 0)
solver.state.H_ext_freq = (4.4e9, 0, 0)

solver.addStepHandler(VTKStorage("switch", "M"), condition.EveryNthStep(20))

log = DataTableLog("vortex-switch.odt")
log.addColumn(("core_x", "core_x"), ("core_y", "core_y"), lambda state: vortex.findCore2(solver, 100e-9, 100e-9))
solver.addStepHandler(log, condition.EveryNthStep(100))

solver.solve(condition.Time(40e-9))
