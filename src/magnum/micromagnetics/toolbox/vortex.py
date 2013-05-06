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

from magnum.mesh import RectangularMesh

import math

__all__ = ["magnetizationFunction", "magnetizationFunction2", "findCore", "findCore2"]

def magnetizationFunction(core_x, core_y, polarization = +1, chirality = +1, core_radius = 10e-9):
    """
    Creates a magnetization function of the form 'lambda state,x,y,z: ...'
    that is suitable to be used with the :func:`magneto.Solver.setM`
    method.

    :param core_x: the core x position in meters
    :param core_y: the core y position in meters
    :param polarization: the vortex polarization, either +1 or -1 (default is +1)
    :param chirality: the vortex chirality, either +1 or -1 (default is +1)
    :param core_radius: a core radius parameter (default is 10e-9 meters)
    :returns: the magnetization function
    """
    def fn(field, pos):
        x, y, z = pos
        Mx = -(y-core_y) * chirality
        My = +(x-core_x) * chirality
        Mz = polarization * core_radius
        scale = 8e5 / math.sqrt(Mx**2 + My**2 + Mz**2)
        return Mx*scale, My*scale, Mz*scale
    return fn

class Skyrmion(object):
    def __init__(self, x, y, p=1, c=1, n=1, r=10e-9):
        self.x = x
        self.y = y
        self.p = p
        self.c = c
        self.n = n
        self.r = r

def magnetizationFunction2(*args):
    def fn(field, pos):
        Mx_ges, My_ges, Mz_ges = 0, 0, 0
        for core in args:
            if isinstance(core, Skyrmion):
                core_x, core_y, polarization, chirality, winding, core_radius = core.x, core.y, core.p, core.c, core.n, core.r
            elif isinstance(core, tuple):
                core_x, core_y, polarization, chirality, winding, core_radius = core
            else:
                raise ValueError("Invalid skyrmion spec")
            x, y, z = pos
            Mx = - winding * (y-core_y) * math.sin(math.pi*chirality/2.0) + (x-core_x) * math.cos(math.pi*chirality/2.0)
            My = + winding * (y-core_y) * math.cos(math.pi*chirality/2.0) + (x-core_x) * math.sin(math.pi*chirality/2.0)
            Mz = polarization * core_radius
            scale = math.exp(-((x-core_x)**2+(y-core_y)**2)/(10*core_radius)**2)/ math.sqrt(Mx**2 + My**2 + Mz**2)
            Mx_ges += Mx*scale
            My_ges += My*scale
            Mz_ges += Mz*scale
        scale_ges = 1.0 / math.sqrt(Mx_ges**2 + My_ges**2 + Mz_ges**2)
        return Mx_ges*scale_ges, My_ges*scale_ges, Mz_ges*scale_ges
    return fn
#magnetizationFunction2(vortex.Skyrmion(x=10e-9, y=20e-9), Skyrmion(...))
#magnetizationFunction((nx*2e-9/2, ny*2e-9/4, 1, 1, 1, 10e-9), (nx*2e-9/2, ny*2e-9*3/4, 1, -1, 1, 10e-9))
#magnetizationFunction((nx*2e-9/2, ny*2e-9/4, 1, 1, 1, 10e-9), (nx*2e-9/2, ny*2e-9/2, 1, 1, -1, 10e-9), (nx*2e-9/2, ny*2e-9*3/4, 1, 1, 1, 10e-9))

def findCore(solver, origin_x = 0e-9, origin_y = 0e-9, body_id = None):
    """
    Tries to determine the vortex core position by finding the absolute
    maximum of absolute value of the z-component of the magnetization. Only
    2d rectangular meshes are supported.  The vortex core is determined
    with sub-cell precision via interpolation.  The returned position can
    be optionally translated by (origin_x, origin_y).

    :param solver: A :class:`magneto.Solver` object. The magnetization that is analyzed
                   for the vortex core detection is taken from the magnetization of the
                   current simulation state of the given solver object.
    :type solver: :class:`magneto.Solver`
    :param origin_x: see above
    :type origin_x: float
    :param origin_y: see above
    :type origin_y: float
    :rtype: 2-tuple of floats
    :param body_id: if given, restrict the search to this body
    :rtype: string
    :returns: the best guess for the vortex core position
    """
    # We only support rectangular meshes
    mesh = solver.world.mesh
    if not isinstance(mesh, RectangularMesh):
        raise ArgumentError("findVortexCore: need world with rectangular mesh!")

    # Get cell indices that occupy the body
    if body_id is not None:
        cell_indices = solver.world.findBody(body_id).shape.getCellIndices(mesh)
    else:
        cell_indices = range(0, mesh.total_nodes)

    nx, ny, nz = mesh.num_nodes
    dx, dy, dz = mesh.delta
    M = solver.state.M

    # Find maximum
    Mz_max = 0
    x, y = 0, 0
    for c in cell_indices:
        Mz = abs(M.get(c)[2])
        if Mz > Mz_max:
            if c >= nx*ny: continue # HACK! for 3d meshes

            Mz_max = Mz
            x = c  % nx
            y = c // nx # // is floor division

    # Refine maximum and convert to meters
    def fit(x0, x1, x2, y0, y1, y2):
        return (-y0*x2**2 + y0*x1**2 + y1*x2**2 - y1*x0**2 - y2*x1**2 + y2*x0**2) / (y0*x1 - y0*x2 - y1*x0 + y1*x2 - y2*x1 + y2*x0) / 2.0

    x2 = dx * (0.5 + fit(x-1, x, x+1, M.get(x-1,y,0)[2], M.get(x,y,0)[2], M.get(x+1,y,0)[2]))
    y2 = dy * (0.5 + fit(y-1, y, y+1, M.get(x,y-1,0)[2], M.get(x,y,0)[2], M.get(x,y+1,0)[2]))
    return x2 - origin_x, y2 - origin_y

# Fast version.
def findCore2(solver, origin_x = 0e-9, origin_y = 0e-9, z_slice=0):
    pos = solver.state.M.findExtremum(z_slice, 2)
    return pos[0] - origin_x, pos[1] - origin_y
