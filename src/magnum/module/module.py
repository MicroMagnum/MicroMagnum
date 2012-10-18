from .assign import assign

class Module(object):
  def calculates(self):
    return []

  def updates(self):
    return []

  def params(self):
    return []

  def initialize(self, system):
    pass

  def properties(self):
    return {}

  def set_param(self, id, value, mask = None):
    p = getattr(self, id)
    p = assign(p, value, mask)
    setattr(self, id, p)

  def get_param(self, id):
    return getattr(self, id)

  def on_param_update(self, id):
    pass

  def name(self):
    return self.__class__.__name__
