class StepHandler(object):
  def __init__(self):
    pass

  def handle(self, state):
    """
    This method is called by the solver when a new simulation step
    needs to be processed by the step handler.  It must be overridden
    by a sub-class.
    """
    raise NotImplementedError("StepHandler.handle(): Implement me!")

  def done(self):
    """
    This method is called by the solver when a simulation is
    complete. Step handlers can use this method to clean up, close log
    files etc.
    """
    pass
