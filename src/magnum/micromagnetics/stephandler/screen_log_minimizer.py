# Copyright 2012, 2013 by the Micromagnum authors.
#
# This file is part of MicroMagnum.
# (at your option) any later version.
#
# MicroMagnum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with MicroMagnum.  If not, see <http://www.gnu.org/licenses/>.

from .screen_log import ScreenLog
#import magnum.tools as tools

import sys

class ScreenLogMinimizer(ScreenLog):
    """
    This step handler produces a log of the minimization on the screen.
    """

    def __init__(self):
        super(ScreenLog, self).__init__(sys.stdout)
        self.addColumn(("step", "step", "", "%d"), lambda state: state.step)
        self.addEnergyColumn("E_tot")
        self.addWallTimeColumn()
        self.addColumn(("deg_per_ns", "deg_per_ns", "deg/ns", "%r"), lambda state: state.deg_per_ns_minimizer)
