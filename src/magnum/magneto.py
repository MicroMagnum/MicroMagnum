try:
  from .magneto_cuda import *
except ImportError:
  from .magneto_cpu import *
