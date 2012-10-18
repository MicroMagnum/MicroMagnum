#!/usr/bin/python
from magnum import *

world = World(
  RectangularMesh((10, 10, 5), (5e-9, 5e-9, 5e-9)),
  Body("freelayer",  Material.Co(),       Cuboid((0e-9, 0e-9, 25e-9), (50e-9, 50e-9, 20e-9))),
  Body("fixedlayer", Material.Co(k1=1e7), Cuboid((0e-9, 0e-9, 15e-9), (50e-9, 50e-9,  0e-9)))
)

p = 1.0, 0.0, 0.0
a_j = -31830

solver = Solver(world, log=True)
solver.setMacroSpinTorque("freelayer", p, a_j)
solver.setZeeman((139260, 0.0, 0.0))
solver.addStepHandler(DataTableStepHandler("macro-spintorque.odt"))

solver.setM((8e5, 8e5, 0))
solver.solve(condition.Time(20e-9))
