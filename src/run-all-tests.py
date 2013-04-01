#!/usr/bin/python

# Copyright 2012, 2013 by the Micromagnum authors.
#
# This file is part of MicroMagnum.
# 
# MicroMagnum is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# MicroMagnum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.

import magnum
from magnum_tests import *

from magnum.config import cfg

import sys
import unittest
if not hasattr(unittest, "skipIf"): # defined in unittest since Python 2.7
  unittest.skipIf = lambda cond, reason: lambda fn: fn
  cfg.skip_long_tests = False
else:
  if "--with-long-tests" in sys.argv:
    cfg.skip_long_tests = False
  elif "--skip-long-tests" in sys.argv:
    cfg.skip_long_tests = True
  else:
    cfg.skip_long_tests = bool(int(input("Skip long tests (0=no,1=yes)? ")))

import os
os.chdir("magnum_tests")
unittest.main(argv=[sys.argv[0]])
