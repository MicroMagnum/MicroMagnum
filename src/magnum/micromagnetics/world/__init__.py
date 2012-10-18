from . import shape

from .material import Material
from .body import Body
from .world import World

__all__ = ["Material", "Body", "World"] + shape.__all__
from .shape import *
