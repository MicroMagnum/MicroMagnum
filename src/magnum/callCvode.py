#!/usr/bin/python
from magnum import *
from math import pi, cos, sin
import magnum.magneto as m
from test import *
from llgDiffEq import *

world = World(RectangularMesh((4,  4, 1), (  5e-9,    5e-9, 3.0e-9)), Body("all", Material.Py(alpha=0.02)))

def state0(field, pos): 
    u = abs(pi*(pos[0]/field.mesh.size[0]-0.5)) / 2.0
    return 8e5 * cos(u), 8e5 * sin(u), 0
# Relax to get initial state for SP4
solver = create_solver(world, [StrayField, ExchangeField], log=True, do_precess=False, evolver="rkf45", eps_abs=1e-4, eps_rel=1e-2)
solver.state.M = state0
solver.state.alpha = 0.5
solver.relax(1.0)

llg = LlgDiffEq(solver.state)

c = MyCvode(llg)
Mydot = solver.state.y
llg.diff(solver.state.y)
#print("")
#c.matrixTest(solver.state.differentiate())
#print("")
#c.matrixTest(solver.state.y)
print("")
c.callPython(llg)
print("")
i = c.cvodeTest()
print("fertig")
print(i)

#d = m.Cvode(solver.state.y, solver.state.differentiate())
#d.cvodeTest()
