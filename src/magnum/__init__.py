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

__version__ = "0.2.0rc"

# I. Import extension lib
from . import magneto

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
  config.cfg.initialize(sys.argv)
do_initialize()
del do_initialize
