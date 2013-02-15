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
import os, sys

## Interactive menus ####################################

if sys.version_info < (3,0):
  getline = raw_input # Python 2.x
else:
  getline = input # Python 3.x

def print_header(header, width):
  pad = (width - len(header)) - 2
  hdr = "="*pad + "[" + header + "]" + "="*pad
  print(hdr)

def interactive_menu(header, text, options):
  print_header(header, 60)
  print(text)
  for idx, opt in enumerate(options):
    print(" %i. %s" % (idx+1, opt))
  while True:
    print("Choice: ", end="")
    try:
      ans = int(getline())
      if ans < 1 or ans > len(options)+1:
        raise ValueError()
    except:
      print("Type a number between 1 and %i." % len(options))
      continue
    break
  return ans

## Fancy colors ########################################

# enable colors if not windows and console is interactive (and thus hopefully supports ansi escape codes)
if os.name != "nt" and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
  def color(c):
    return "\033[" + str(30+c) + "m"
  def nocolor():
    return "\033[0m"
else:
  def color(c):
    return ""
  def nocolor():
    return ""
