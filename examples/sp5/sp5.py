#!/usr/bin/python
from magnum import *

############################################################
# Create a world                                           #
############################################################

#world = World(RectangularMesh((40, 40, 1), (2.5e-9, 2.5e-9, 10e-9)), Body("all", Material.Py(xi=0.05, P=1.0, alpha=0.1)))
world = World(RectangularMesh((100, 100, 1), (1e-9, 1e-9, 10e-9)), Body("all", Material.Py(xi=0.05, P=1.0, alpha=0.1)))

############################################################
# First step: Relax to get a vortex state                  #
############################################################

solver = create_solver(world, [StrayField, ExchangeField], log=True)
solver.state.M = vortex.magnetizationFunction(core_x=50e-9, core_y=50e-9, polarization=1, core_radius=10e-9)
solver.state.alpha = 0.3
solver.relax()
vortex_M = solver.state.M  # save vortex pattern in 'vortex_M' vector field

writeOMF("sp5-M0.omf", vortex_M) # save start configuration

############################################################
# Second step: Apply DC to vortex                          #
############################################################

solver = create_solver(world, [StrayField, ExchangeField, SpinTorque, AlternatingCurrent], log=True)

solver.state.M = vortex_M
solver.state.j_offs = (1e12, 0, 0)

# record vortex core position in .odt file "sp5.odt"
odt = DataTableLog("sp5-100-100.odt")
odt.addColumn(("Vx", "Vortexcore X-pos.", "m"), ("Vy", "Vortexcore Y-pos.", "m"), lambda s: vortex.findCore(solver, 50e-9, 50e-9, "all"))
solver.addStepHandler(odt, condition.EveryNthStep(100))

solver.solve(condition.Time(30.0e-9)) # start simulation

writeOMF("sp5-M_end.omf", solver.state.M) # save end configuration
