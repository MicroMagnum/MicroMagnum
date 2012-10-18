#!/usr/bin/python
from magnum import *

# Simple Larmor precession frequency test.
# Analytical frequency: MU0*H = 2.2102e11 rad/s
mesh = RectangularMesh((1,1,1), (1e-9, 1e-9, 1e-9))
mat  = Material.Py(alpha=0, Ms=1.0/MU0)
world = World(mesh, Body("all", mat, Everywhere()))

solver = create_solver(world, [ExternalField], log=True, eps_rel=0.0000000001)
solver.state.M = (1,1,1)
solver.state.H_ext_offs = (0,0,1e6)
solver.addStepHandler(DataTableLog("larmor.odt"), condition.Always())
solver.solve(condition.Time(0.3e-9))
