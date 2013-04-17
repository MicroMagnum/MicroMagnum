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

from math import sin, cos

# Helper class: Handles some affine transformations in 3-D using 4-D matrices
# ( see e.g. http://en.wikipedia.org/wiki/Transformation_matrix )
class Transformation(object):
    def __init__(self):
        self.T = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]

    def addTransform(self, T2):
        T1 = self.T
        T3 = [[0,0,0,0],
              [0,0,0,0],
              [0,0,0,0],
              [0,0,0,0]]
        # Matrix multiply: T3 = T1 * T2
        for i in range(4):
            for j in range(4):
                T3[i][j] = T2[i][0]*T1[0][j] + T2[i][1]*T1[1][j] + T2[i][2]*T1[2][j] + T2[i][3]*T1[3][j]
        self.T = T3

    def addScale(self, sx, sy, sz):
        self.addTransform(((sx,0,0,0),
                           (0,sy,0,0),
                           (0,0,sz,0),
                           (0,0,0, 1)))

    def addTranslate(self, tx, ty, tz):
        self.addTransform(((1,0,0,tx),
                           (0,1,0,ty),
                           (0,0,1,tz),
                           (0,0,0,1)))

    def addRotate(self, a, u):
        # a: angle, u: unit vector (rotation axis)
        self.addTransform(((cos(a)+u[0]**2*(1-cos(a)), u[0]*u[1]*(1-cos(a))-u[2]*sin(a), u[0]*u[2]*(1-cos(a))+u[1]*sin(a),0),
                           (u[1]*u[0]*(1-cos(a))+u[2]*sin(a), cos(a)+u[1]**2*(1-cos(a)), u[1]*u[2]*(1-cos(a))-u[0]*sin(a),0),
                           (u[2]*u[0]*(1-cos(a))-u[1]*sin(a), u[2]*u[1]*(1-cos(a))+u[0]*sin(a), cos(a)+u[2]**2*(1-cos(a)),0),
                           (0,0,0,1)))

    def transformPoint(self, p1): # p1 = (a, b, c)
        T1 = self.T
        p2 = [0,0,0,0]
        for i in range(4):
            p2[i] = T1[i][0]*p1[0] + T1[i][1]*p1[1] + T1[i][2]*p1[2] + T1[i][3]
        return (p2[0]/p2[3],p2[1]/p2[3],p2[2]/p2[3])
