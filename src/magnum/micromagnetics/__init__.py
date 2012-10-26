# Copyright 2012 by the Micromagnum authors.
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

## module system and solver for micromagnetics
from .micro_magnetics import MicroMagnetics
from .micro_magnetics_solver import MicroMagneticsSolver
from .shortcuts import create_solver

## constants and modules
from .constants import MU0, H_BAR, ELECTRON_CHARGE, MU_BOHR, GYROMAGNETIC_RATIO, BOLTZMANN_CONSTANT
from .landau_lifshitz_gilbert import LandauLifshitzGilbert
from .exchange_field import ExchangeField
from .stray_field import StrayField, StrayFieldCalculator
from .external_field import ExternalField
from .anisotropy_field import AnisotropyField
from .homogeneous_field import HomogeneousField, HomogeneousCurrent
from .spin_torque import SpinTorque
#from .macro_spin_torque import MacroSpinTorque
from .alternating_field import AlternatingField
from .alternating_current import AlternatingCurrent
from .simple_field import SimpleExternalField, SimpleVectorField

__all__ = [
	"MicroMagnetics", "MicroMagneticsSolver", "create_solver",
	"MU0", "H_BAR", "ELECTRON_CHARGE", "MU_BOHR", "GYROMAGNETIC_RATIO", "BOLTZMANN_CONSTANT",
	"LandauLifshitzGilbert", "ExchangeField", "StrayField", "StrayFieldCalculator",
	"ExternalField", "AnisotropyField", "HomogeneousField", "HomogeneousCurrent",
	"SpinTorque", "AlternatingField", "AlternatingCurrent", "SimpleExternalField", "SimpleVectorField"
]

## submodules
from . import io
from . import world
from . import stephandler
from . import toolbox
from . import condition

__all__.extend(world.__all__ + stephandler.__all__ + toolbox.__all__ + io.__all__ + condition.__all__)
from .world import *
from .stephandler import *
from .toolbox import *
from .io import *
from .condition import *
