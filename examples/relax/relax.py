#!/usr/bin/python
from magnum import *
import sys

omf_file = "test.omf"
mag0 = readOMF(omf_file)

world = World(mag0.mesh, Material.Py())
solver = create_solver(world, [StrayField, ExchangeField], log=True)
solver.state.alpha = 0.5
solver.state.M = mag0

print("Loaded %s, evolving until relaxed." % omf_file)
print("I assume that the material is permalloy.")

solver.relax(1.0)
writeOMF("relaxed_" + omf_file, solver.state.M)
