#!/usr/bin/python
from magnum import *
import sys

if sys.version_info < (3,0):
    path = raw_input("Enter magnetization (.omf file format): ")
else:
    path = input("Enter magnetization (.omf file format): ")


M = readOMF(path)
H = VectorField(M.mesh)

sfc = StrayFieldCalculator(M.mesh)
sfc.calculate(M, H)

import re
writeOMF(re.sub("\\.omf$", "", path) + "-H_stray.ohf", H)
