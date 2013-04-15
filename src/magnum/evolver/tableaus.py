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

import magnum.magneto as magneto

def __assemble_butcher_tableau(num_steps, tab_a, tab_b, tab_c, tab_ec):
    assert len(tab_a) == num_steps and len(tab_c) == num_steps and len(tab_ec) == num_steps

    # I. Assemble tableau
    tab = magneto.ButcherTableau(num_steps)
    # sampling points
    for i, a_i in enumerate(tab_a):
        tab.setA(i, a_i)
    # coefficients b_ij
    for i, row in enumerate(tab_b):
        for j, b_ij in enumerate(row):
            tab.setB(i, j, b_ij)
    # embedded order coeffs
    for i, c_i in enumerate(tab_c):
        tab.setC(i, c_i)
    # these are the differences of fifth and fourth order coefficients for error estimation
    for i, ec_i in enumerate(tab_ec):
        tab.setEC(i, ec_i)

    # II. Some verification (incomplete..)
    s1, s2 = 0.0, 0.0
    for i in range(num_steps):
        s1 += tab.getC (i)
        s2 += tab.getEC(i)
    assert abs(1.0 - s1) < 1e-7 and abs(s2) < 1e-7

    for i in range(num_steps):
        sum = 0.0
        for j in range(i):
            sum += tab.getB(i, j)
        assert abs(sum - tab.getA(i)) < 1e-7

    return tab

def rkf45(): # Runge-Kutta-Fehlberg
    # Values taken from libgsl
    # tab_a  = [0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0]
    # tab_b  = [[],
    #           [1.0/4.0],
    #           [3.0/32.0, 9.0/32.0],
    #           [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0],
    #           [8341.0/4104.0, -32832.0/4104.0, 29440.0/4104.0, -845.0/4104.0],
    #           [-6080.0/20520.0, 41040.0/20520.0, -28352.0/20520.0, 9295.0/20520.0, -5643.0/20520.0]]
    # tab_c  = [902880.0/7618050.0, 0.0, 3953664.0/7618050.0, 3855735.0/7618050.0, -1371249.0/7618050.0, 277020.0/7618050.0]
    # tab_ec = [1.0/360.0, 0.0, -128.0/4275.0, -2197.0/75240.0, 1.0/50.0, 2.0/55.0]

    # Values taken from Wikipedia (en)
    tab_a  = [0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0]
    tab_b  = [[],
              [1.0/4.0],
              [3.0/32.0, 9.0/32.0],
              [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0],
              [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0],
              [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0]]
    tab_c4 = [25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0]              # fourth-order
    tab_c5 = [16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0]     # fifth-order
    tab_c  = tab_c4 # Advance with 4th-order method
    tab_ec = [tab_c4[i] - tab_c5[i] for i in range(6)]

    tab = __assemble_butcher_tableau(6, tab_a, tab_b, tab_c, tab_ec)
    tab.fsal = False
    tab.order = 4
    return tab

def cc45(): # Cash'n'Karp
    # Values taken from Wikipedia (en)
    tab_a  = [0.0, 1.0/5.0, 3.0/10.0, 3.0/5.0, 1.0, 7.0/8.0]
    tab_b  = [[],
              [1.0/5.0],
              [3.0/40.0, 9.0/40.0],
              [3.0/10.0, -9.0/10.0, 6.0/5.0],
              [-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0],
              [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]]
    tab_c4 = [2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 1.0/4.0]  # fourth-order
    tab_c5 = [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]                   # fifth-order
    tab_c  = tab_c4 # Advance with 4th-order method
    tab_ec = [tab_c4[i] - tab_c5[i] for i in range(6)]

    tab = __assemble_butcher_tableau(6, tab_a, tab_b, tab_c, tab_ec)
    tab.fsal = False
    tab.order = 4
    return tab

def dp54(): # Dormand-Prince
    # Values taken from Wikipedia (en)
    tab_a = [0.0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0]
    tab_b = [[],
             [1.0/5.0],
             [3.0/40.0, 9.0/40.0],
             [44.0/45.0, -56.0/15.0, 32.0/9.0],
             [19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0],
             [9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0],
             [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0]]
    tab_c4 = [5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0] # fourth-order
    tab_c5 = [35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0]                  # fifth-order
    tab_c  = tab_c5 # Advance with fifth-order method
    tab_ec = [tab_c4[i] - tab_c5[i] for i in range(7)]

    tab = __assemble_butcher_tableau(7, tab_a, tab_b, tab_c, tab_ec)
    tab.fsal = True
    tab.order = 5
    return tab

def rk23(): # Bogacki-Shampine
    # Values taken from http://en.wikipedia.org/wiki/Bogacki-Shampine
    tab_a = [0.0, 0.5, 3.0/4.0, 1.0]
    tab_b = [[],
             [1.0/2.0],
             [0.0    , 3.0/4.0],
             [2.0/9.0, 1.0/3.0, 4.0/9.0]]
    tab_c2 = [7.0/24.0, 1.0/4.0, 1.0/3.0, 1.0/8.0] # second-order
    tab_c3 = [2.0/9.0 , 1.0/3.0, 4.0/9.0, 0.0    ] # third-order

    tab_c  = tab_c3 # advance with third-order method
    tab_ec = [tab_c2[i] - tab_c3[i] for i in range(4)]

    tab = __assemble_butcher_tableau(4, tab_a, tab_b, tab_c, tab_ec)
    tab.fsal = True
    tab.order = 3
    return tab
