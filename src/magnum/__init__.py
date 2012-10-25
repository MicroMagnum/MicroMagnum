# Copyright 2012 by the Micromagnum authors.
#
# This file is part of MicroMagnum.
# 
# Foobar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Foobar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

# I. Import extension lib
from . import magneto

__version__ = "0.2.0rc"

import sys

# II. Import config & tools
from . import tools
from . import console
from . import config
from . import module
from . import evolver

# III. Import submodules
from . import controller
from . import mesh
from . import solver
from . import micromagnetics

# V. Prepare exports for "from magnum import *"
__all__ = controller.__all__ + mesh.__all__ + solver.__all__ + micromagnetics.__all__
from .controller import *
from .mesh import *
from .solver import *
from .micromagnetics import *

# VI. Initialize micromagnum via the main config object
def do_initialize():
  import sys
  # No nothing when magnum is imported for documentation generation
  if 'sphinx-build' in sys.argv[0]: return
  # Filter out --skip-long-tests and --with-long-tests parameters
  argv = list(filter(lambda p: p != '--skip-long-tests' and p != '--with-long-tests', sys.argv))
  # Initialize MicroMagnum
  config.cfg.initialize(argv)
do_initialize()
del do_initialize

# mesh0 = RectangularMesh((100,100,1),(1e-9,1e-9,1e-9))
# mesh1 = RectangularMesh((200,200,1),(1e-9,1e-9,1e-9))
# mesh2 = RectangularMesh((300,300,1),(1e-9,1e-9,1e-9))
