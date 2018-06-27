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

from __future__ import print_function

import sys
import os
import itertools


def flush():
    import gc
    from magnum.magneto import flush
    gc.collect()
    flush()


def cpu_count():
    try:
        import multiprocessing
        return multiprocessing.cpu_count()
    except:
        from magnum.logger import logger
        logger.warn("Could not find out number of processors")
        return 1

## Fancy colors #########################################

# Enable colors if not windows and console is interactive (and thus
# hopefully supports ansi escape codes)
# TODO: Maybe check $TERM variable.
if os.name != "nt" and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
    def color(c):
        return "\033[" + str(30 + c) + "m"

    def nocolor():
        return "\033[0m"
else:
    def color(c):
        return ""

    def nocolor():
        return ""

## Portable xrange ######################################

if sys.version_info < (3, 0):
    irange = xrange
else:
    irange = range

## Interactive menus ####################################

if sys.version_info < (3, 0):
    getline = raw_input
else:
    getline = input


def print_header(header, width):
    pad = (width - len(header)) - 2
    hdr = "=" * pad + "[" + header + "]" + "=" * pad
    print(hdr)


def interactive_menu(header, text, options):
    print_header(header, 60)
    print(text)
    for idx, opt in enumerate(options):
        print("  %i. %s" % (idx + 1, opt))
    while True:
        print("Choice: ", end="")
        sys.stdout.flush()
        try:
            ans = int(getline())
            if ans < 1 or ans > len(options) + 1:
                raise ValueError()
        except:
            print("Type a number between 1 and %i." % len(options))
            continue
        break
    return ans


## Generate a list of floats from a range ###############

def frange(*args):
    """
    A float range generator. Usage:

    frange(start [,stop [,step]]) -> generator

    Examples:
      list(frange(4.2))            -> [0.0, 1.0, 2.0, 3.0, 4.0]
      list(frange(2.2, 5.6))       -> [2.2, 2.3, 4.3, 5.3]
      list(frange(2.2, 5.6, 0.25)) -> [2.2, 2.45, 2.7, 2.95, 3.2, 3.45, 3.7, 3.95, 4.2, 4.45, 4.7, 4.95, 5.2, 5.45]
    """
    start = 0.0
    step = 1.0

    l = len(args)
    if l == 1:
        end, = args
    elif l == 2:
        start, end = args
    elif l == 3:
        start, end, step = args
        if step == 0.0:
            raise ValueError("step must not be zero")
    else:
        raise TypeError("frange expects 1-3 arguments, got %d" % l)

    v = start
    while True:
        if (step > 0 and v >= end) or (step < 0 and v <= end):
            raise StopIteration
        yield v
        v += step


def range_2d(nx, ny):
    return itertools.product(range(nx), range(ny))


def range_3d(nx, ny, nz):
    return itertools.product(range(nx), range(ny), range(nz))


def makedirs(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            # Sometimes many parallel MicroMagNum are started at
            # the same time. Another process might have created
            # the directory already, so that the os.makedirs call
            # above fails. In this case, we can savely ignore the
            # exception if the path now exists.
            if not os.path.exists(path):
                raise
    return path
