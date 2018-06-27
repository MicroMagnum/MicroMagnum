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

from magnum.micromagnetics.alternating_field import AlternatingField
from magnum.micromagnetics.static_field import StaticField

class AlternatingCurrent(AlternatingField):
    def __init__(self, var_id = "j"):
        super(AlternatingCurrent, self).__init__(var_id)

class StaticCurrent(AlternatingField):
    def __init__(self, var_id = "j"):
        super(StaticCurrent, self).__init__(var_id)
