#!/usr/bin/python
import magnum
from magnum.config import cfg

import sys
if   "--with-long-tests" in sys.argv:
  cfg.skip_long_tests = False
elif "--skip-long-tests" in sys.argv:
  cfg.skip_long_tests = True
else:
  cfg.skip_long_tests = bool(int(input("Skip long tests (0=no,1=yes)? ")))
argv = list(filter(lambda p: p != '--with-long-tests' and p != '--skip-long-tests', sys.argv))

# include all tests here...
from controller import *
from evolver import *
from mesh import *
from world import *
from magneto import *
from modules import *
unittest.main(argv = argv)
