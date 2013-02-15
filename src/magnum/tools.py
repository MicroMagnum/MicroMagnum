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

from __future__ import absolute_import

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
        end = args[0]
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

def flush():
  import gc
  gc.collect()

  from .magneto import flush
  flush()

  #if len(gc.garbage) > 0:
  #  logger.warn("Uncollectable garbage!")

def cpu_count():
  try:
    import multiprocessing
    return multiprocessing.cpu_count()
  except:
    logger.warn("Could not find out number of processors")
    return 1
