from .alternating_field import AlternatingField

class AlternatingCurrent(AlternatingField):
  def __init__(self, var_id = "j"):
    super(AlternatingCurrent, self).__init__(var_id)
