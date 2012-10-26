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

from .exchange import ExchangeTest
from .slonchewski_test import FDMSlonchewskiTest
from .spintorque_test import FDMZhangLiTest
from .anisotropy_test import UniaxialAnisotropyTest, CubicAnisotropyTest
from .llge_test import LLGETest
from .matrix_test import MatrixTest, VectorMatrixTest
from .scaled_abs_max_test import ScaledAbsMaxTest
from .stray_field_test import StrayFieldTest
from .numpy_interaction_test import NumpyInteractionTest
