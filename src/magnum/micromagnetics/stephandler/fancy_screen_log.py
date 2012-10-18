from __future__ import print_function

from magnum.solver import StepHandler

import magnum.console as console

import sys

class FancyScreenLog(StepHandler):
  def __init__(self):
    super(FancyScreenLog, self).__init__()

  info = [
    ("Step number..... : %i", lambda state: state.step),
    ("Time............ : %2.6f ns", lambda state: state.t * 1e9),
    ("Step size....... : %2.6f ns", lambda state: state.h * 1e9),
    ("<Mx>,<My>,<Mz>.. : %8.2f, %8.2f, %8.2f", lambda state: state.M.average())
  ]

  def handle(self, state):
    sys.stdout.write("\033[s") # save cursor position
    sys.stdout.write("\033[H") # Cursor to top left corner

    erase_line = "\033[2K"
 
    print(erase_line + "+================================================================+")
    for fmt, fn in FancyScreenLog.info:
      print(erase_line + fmt % fn(state))
    print(erase_line + "+================================================================+")
    
    sys.stdout.write("\033[u") # restore cursor position

  def done(self):
    pass
