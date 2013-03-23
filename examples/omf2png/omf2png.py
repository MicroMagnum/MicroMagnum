#!/usr/bin/python
from magnum import *

import sys, os
if len(sys.argv) != 2:
  print "Usage: %s <input.omf>" % sys.argv[0]
  sys.exit(-1)
omf_file = sys.argv[1]
base, ext = os.path.splitext(omf_file)

M = readOMF(omf_file)
writeImage(base + "-x.png", M, "x")
writeImage(base + "-y.png", M, "y")
writeImage(base + "-z.png", M, "z")
writeImage(base + "-xy-angle.png", M, "xy-angle")
writeImage(base + "-mag.png", M, "mag")
