#!/usr/bin/python
from magnum import *

import sys, os
if len(sys.argv) != 2:
  print "Usage: %s <input.omf>" % sys.argv[0]
  sys.exit(-1)
omf_file = sys.argv[1]
base, ext = os.path.splitext(omf_file)

M = readOMF(omf_file)

Ms = M.absMax()

writeImage(base + "-x.png", M, "x", colorrange = (-Ms, +Ms))
writeImage(base + "-y.png", M, "y", colorrange = (-Ms, +Ms))
writeImage(base + "-z.png", M, "z", colorrange = (-Ms, +Ms))
writeImage(base + "-xy-angle.png", M, "xy-angle", colorrange = (-math.pi, +math.pi))
writeImage(base + "-mag.png", M, "mag", colorrange = (0, Ms))
