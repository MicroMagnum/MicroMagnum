#import math

class Condition(object):
  def __init__(self, fn = None):
    """
    Construct a new Condition object.

    :param fn: a function of the form 'lambda state: ...' that checks whether the condition 
               is true for a magnetic state. *The function should have no side-effects*.
    """
    self.fn = fn

  def check(self, state):
    """
    Returns whether this condition applies to a given magnetic state.

    :param state: the magnetic state that the condition is being checked against.
    :rtype: bool
    """
    return self.fn(state)

  def get_time_of_interest(self, state):
    """
    Returns either None or a time in the future (regarding state.t). By
    returning a time, the condition indicates that it wants to check a
    state at this time.
    """
    return None

  def __and__(self, other): # and
    """
    Combines this condition with the other condition. 
    The returned condition is true when both conditions yield true.
    This operator is invoked with the 'and' operator.

    :param other: the other condition
    :type other: :class:`magneto.Condition`
    :rtype: :class:`magneto.Condition`
    :returns: a newly created condition
    """
    return AndCondition(self, other) 

  def __or__(self, other): # or
    """
    Combines this condition with the other condition. 
    The returned condition is true when at least one condition yields true.
    This operator is invoked with the 'or' operator.

    :param other: the other condition
    :type other: :class:`magneto.Condition`
    :rtype: :class:`magneto.Condition`
    :returns: a newly created condition
    """
    return OrCondition(self, other) 

  def __invert__(self): # ~
    """
    Creates a new condition, which is true if and only if this conditon is false.
    This operator is invoked with the '~' operator.

    :param other: the condition to invert
    :type other: :class:`magneto.Condition`
    :rtype: :class:`magneto.Condition`
    :returns: a newly created condition
    """
    return InvertCondition(self)

class AndCondition(Condition):
  def __init__(self, op1, op2):
    self.__op1, self.__op2 = op1, op2
    super(AndCondition, self).__init__(lambda state: op1.check(state) and op2.check(state))

  def get_time_of_interest(self, state):
    t = min([self.__op2.get_time_of_interest(state) or 1e100, self.__op1.get_time_of_interest(state) or 1e100])
    if t == 1e100: return None
    return t

class OrCondition(Condition):
  def __init__(self, op1, op2):
    self.__op1, self.__op2 = op1, op2
    super(OrCondition, self).__init__(lambda state: op1.check(state) or op2.check(state))

  def get_time_of_interest(self, state):
    t = min([self.__op2.get_time_of_interest(state) or 1e100, self.__op1.get_time_of_interest(state) or 1e100])
    if t == 1e100: return None
    return t

class InvertCondition(Condition):
  def __init__(self, op1):
    self.__op1 = op1
    super(InvertCondition, self).__init__(lambda state: not op1.check(state))

  def get_time_of_interest(self, state):
    return self.__op1.get_time_of_interest(state)

class EveryNthStep(Condition):
  def __init__(self, nth):
    """
    Condition that is true at every *n*-th simulation step.

    :param nth: specifies *every 'nth' simulation step*
    :type nth: int
    """
    super(EveryNthStep, self).__init__(lambda state: state.step % nth == 0)

class EveryNthSecond(Condition):
  def __init__(self, nth):
    super(EveryNthSecond, self).__init__()
    self.__nth = nth

  def check(self, state):
    dt = state.t % self.__nth # float module: distance of state.t to multiples of "nth".
    #                         Fix for floating point arithmetics
    return abs(dt) < 1e-16 or abs(dt - self.__nth) < 1e-16

  def get_time_of_interest(self, state):
    nth = self.__nth
    return (int(state.t / nth)+1) * nth

class AfterNthStep(Condition):
  def __init__(self, nth):
    """
    Condition that is true after each simulation step after the n-th simulation step.

    :param nth: specifies *after 'nth' simulation step*
    :type nth: int
    """
    super(AfterNthStep, self).__init__(lambda state: state.step > nth)

class TimeGreaterEq(Condition):
  def __init__(self, t_max):
    """
    Condition that is true after the simulation time 't_max' is reached.

    :param t_max: specifies the simulation time
    :type t_max: float
    :rtype: :class:`magneto.Condition`
    :returns: a newly created condition
    """
    return super(TimeGreaterEq, self).__init__(lambda state: state.t >= t_max)

Time = TimeGreaterEq # alias

class TimeBetween(Condition):
  def __init__(self, t0, t1):
    """
    Condition that is true between the simulation time interval [t0, t1].

    :param t0: specifies the left border of the time interval in seconds
    :param t1: specifies the right border of the time interval in seconds
    """
    return super(TimeBetween, self).__init__(lambda state: state.t >= t0 and state.t < t1)

class Relaxed(Condition):
  def __init__(self, max_degree_per_ns = 1, check_every_nth_step = 100):
    """
    Returns condition that is true when the time deriviative of the magnetization is small, i.e.
    when the magnetic state is relaxed.
    :param max_degree_per_ns: Maximum allowed magnetization change in degrees per nanosecond to consider the state relaxed.
    :param check_every_nth_step: Check only every nth step (to save computation time)
    """
    #def test(state):
    #  if state.step % check_every_nth_step != 0: return False
    #  deg_per_timestep = (180.0 / math.pi) * math.atan2(state.dMdt.absMax() * state.h, state.M.absMax()) # we assume a<b at atan(a/b).
    #  deg_per_ns = 1e-9 * deg_per_timestep / state.h 
    #  return deg_per_ns <= max_degree_per_ns

    def test(state):
      # Only check everty nth step.
      if state.step % check_every_nth_step != 0:
        return False

      # Calculate magnetization change in degree per ns.
      deg_per_ns = state.deg_per_ns

      # Remember last degrees per ns.
      try:
        last_deg_per_ns = getattr(state, "last_deg_per_ns_%s" % id(self))
      except AttributeError:
        last_deg_per_ns = None
      setattr(state, "last_deg_per_ns_%s" % id(self), deg_per_ns)

      if last_deg_per_ns is None: return False # Not enough data? Bail out.

      # Now we have valid values in 'deg_per_ns' and 'last_deg_per_ns'.
      # The state is considered relaxed if degrees per ns is small enough (< max_degree_per_ns)
      # and became smaller during the last time step (< last_deg_per_ns).
      is_relaxed = deg_per_ns < last_deg_per_ns and deg_per_ns <= max_degree_per_ns
      return is_relaxed

    return super(Relaxed, self).__init__(test)

class Always(Condition):
  def __init__(self):
    """
    Condition that is always true.
    """
    return super(Always, self).__init__(lambda state: True)

class Never(Condition):
  def __init__(self):
    """
    Condition that is never true.
    """
    return super(Never, self).__init__(lambda state: False)

class Once(Condition):
  def __init__(self, cond):
    tag = "_Condition_once_%s" % id(self) # Generate a (hopefully) unique tag to attach to state.
    def test(state):
      if not hasattr(state, tag) and cond.check(state):
        setattr(state, tag, True)
        return True
      return False

    return super(Once, self).__init__(test)
