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

## solver for micromagnetics
from magnum.micromagnetics.create_solver import create_solver

## constants and modules
from .constants import MU0, H_BAR, ELECTRON_CHARGE, MU_BOHR, GYROMAGNETIC_RATIO, BOLTZMANN_CONSTANT
from .landau_lifshitz_gilbert import LandauLifshitzGilbert
from .exchange_field import ExchangeField
from .stray_field import StrayField, StrayFieldCalculator
from .external_field import ExternalField
from .anisotropy_field import AnisotropyField
from .homogeneous_field import HomogeneousField, HomogeneousCurrent
from .spin_torque import SpinTorque
from .alternating_field import AlternatingField
from .alternating_current import AlternatingCurrent
from .simple_field import SimpleExternalField, SimpleVectorField
from .current_path import CurrentPath

__all__ = [
    "create_solver",
    "MU0", "H_BAR", "ELECTRON_CHARGE", "MU_BOHR", "GYROMAGNETIC_RATIO", "BOLTZMANN_CONSTANT",

    "LandauLifshitzGilbert",
    "ExchangeField",
    "StrayField", "StrayFieldCalculator",
    "AnisotropyField",
    "AlternatingExternalField", "StaticExternalField", "ExternalField",
    "SpinTorque",
    "AlternatingCurrent", "StaticCurrent",
    "AlternatingField",
    "StaticField",
    "CurrentPath"
]
