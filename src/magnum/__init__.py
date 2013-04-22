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

__version__ = "0.2rc4"


# I. Import public API of modules in this package
from magnum.tools import flush, frange, irange, range_3d
__all__ = ["flush", "frange", "irange", "range_3d"]


# II. Import subpackages
import magnum.controller
import magnum.mesh
import magnum.module
import magnum.solver
import magnum.evolver
import magnum.micromagnetics
import magnum.micromagnetics.io
import magnum.micromagnetics.world
import magnum.micromagnetics.stephandler
import magnum.micromagnetics.toolbox


# III. Merge public API into magnum namespace.
def merge_package(src):
    for var in src.__all__:
        magnum.__all__.append(var)
        setattr(magnum, var, getattr(src, var))

merge_package(magnum.controller)
merge_package(magnum.mesh)
merge_package(magnum.solver)
merge_package(magnum.micromagnetics)
merge_package(magnum.micromagnetics.io)
merge_package(magnum.micromagnetics.world)
merge_package(magnum.micromagnetics.stephandler)
merge_package(magnum.micromagnetics.toolbox)

del merge_package


# III. Initialize MicroMagnum via the configuration object
import sys
import atexit
if 'sphinx-build' in sys.argv[0]:
    # No nothing when magnum is imported for documentation generation
    pass
else:
    import magnum.config
    magnum.config.initialize(sys.argv)
    atexit.register(magnum.config.deinitialize)
del sys
del atexit
