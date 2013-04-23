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

# StepHandler [abstract]
#  |- LogStepHandler [abstract]
#  |   |- ScreenLog
#  |   |- DataTableLog
#  |- StorageStepHandler [abstract]
#  |   |- VTKStorage
#  |   |- OOMMFStorage
#  |   |- ImageStorage
#  |- FancyScreenLog

from magnum.micromagnetics.stephandler.oommf_storage import OOMMFStorage
from magnum.micromagnetics.stephandler.image_storage import ImageStorage
from magnum.micromagnetics.stephandler.vtk_storage import VTKStorage
from magnum.micromagnetics.stephandler.screen_log import ScreenLog
from magnum.micromagnetics.stephandler.data_table_log import DataTableLog
from magnum.micromagnetics.stephandler.fancy_screen_log import FancyScreenLog
from magnum.micromagnetics.stephandler.web import WebStepHandler

__all__ = [
    "OOMMFStorage", "ImageStorage", "VTKStorage", "ScreenLog",
    "DataTableLog", "FancyScreenLog", "WebStepHandler"
]
