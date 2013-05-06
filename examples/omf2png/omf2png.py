#!/usr/bin/python
import sys
import os
import math

from magnum import readOMF, writeImage

if len(sys.argv) != 2:
    print "Usage: %s <input.omf>" % sys.argv[0]
    sys.exit(-1)
omf_file = sys.argv[1]
base, ext = os.path.splitext(omf_file)

M = readOMF(omf_file)

Ms = M.absMax()

writeImage(base + "-x.png", M, "x", color_range=(-Ms, +Ms))
writeImage(base + "-y.png", M, "y", color_range=(-Ms, +Ms))
writeImage(base + "-z.png", M, "z", color_range=(-Ms, +Ms))
writeImage(base + "-xy-angle.png", M, "xy-angle", color_range=(-math.pi, +math.pi))
writeImage(base + "-mag.png", M, "mag", color_range=(0, Ms))
