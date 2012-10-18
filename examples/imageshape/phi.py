#!/usr/bin/python
from magnum import *

mesh = RectangularMesh((300,300,1), (5e-9, 5e-9, 60e-9))
isc = ImageShapeCreator("phi.png", mesh)
world = World(mesh, Body("phi", Material.Py(), isc.pick("black")))

solver = create_solver(world, [StrayField, ExchangeField], log=True)
solver.state["phi"].M = (8e5,0,0)

writeOMF("phi.omf", solver.state.M)

solver.relax()

writeOMF("phi_end.omf", solver.state.M)
