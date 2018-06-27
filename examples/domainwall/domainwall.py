#!/usr/bin/python
from magnum import *

world = World(
  RectangularMesh((250, 25, 1), (4e-9, 4e-9, 4e-9)),
  Body("lt", Material.Py(), Cuboid((  0e-9, 0e-9, 0e-9), ( 400e-9, 100e-9, 4e-9))),
  Body("ct", Material.Py(), Cuboid((400e-9, 0e-9, 0e-9), ( 600e-9, 100e-9, 4e-9))),
  Body("rt", Material.Py(), Cuboid((600e-9, 0e-9, 0e-9), (1000e-9, 100e-9, 4e-9)))
)

# Relax a domain wall
solver = create_solver(world, [StrayField, ExchangeField], log=True)
solver.state.alpha = 0.5
solver.state["lt"].M = (-8e5,   0, 0) # left side: left
solver.state["ct"].M = (   0, 8e5, 0) # center: up
solver.state["rt"].M = ( 8e5,   0, 0) # right side: right
solver.relax()
mag = solver.state.M

# Apply current
j0 = (4e11, 0, 0)
solver = create_solver(world, [StrayField, ExchangeField, SimpleVectorField("j"), SpinTorque], log=True)
solver.state.M = mag
solver.state.j = j0
solver.addStepHandler(DataTableLog("domainwall.odt"), condition.EveryNthStep(20))
solver.addStepHandler(OOMMFStorage("domainwall", ["M", "H_stray"]), condition.EveryNthStep(20))
solver.solve(condition.Time(2e-9))


writeOMF("DomainWall.omf", solver.state.M)

