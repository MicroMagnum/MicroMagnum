#!/usr/bin/python
from magnum import *

world = World(
  RectangularMesh((100, 100, 1), (2e-9, 2e-9, 1e-8)),
  Material.Py()
)

solver = create_solver(world, [StrayField, ExchangeField], log=True)
solver.state.M = 8e5,0,0
solver.relax(5)

writeVTK("relaxed-state.vtr", solver.state.M)
writeOMF("relaxed-state.omf", solver.state.M)
