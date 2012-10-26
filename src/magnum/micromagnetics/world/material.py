# Copyright 2012 by the Micromagnum authors.
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

from magnum.logger import logger

class Material(object):
  def __init__(self, params):
    """
    Create a new material object. 'params' must be a dict that maps
    material parameter string-ids to their values.
    """
    self.__params = params

  def get(self, key):
    """
    Return the material parameter with id-string 'key'. The material parameter
    is usually a scalar value.
    """
    return self.__params[key]

  def __getattr__(self, key):
    try:
      return self.__params[key]
    except:
      raise AttributeError("Material instance has no attribute '%s'" % key)

  def __repr__(self):
    return "Material(" + repr(self.__params) + ")"

  @staticmethod
  def fromDB(id, **params):
    """
    Return a material object of material 'id' from the database. E.g.: Material.fromDB("Py").
    Additional keyword options are merged with the material parameters of the newly created instance.
    """
    db_params = Material.__getDBParameters(id)
    return Material(dict(list(db_params.items()) + list(params.items())))
    
  @staticmethod
  def Py(**params):
    """
    Additional keyword options are merged with the material parameters of the newly created instance.
    Return a material object for the material 'Permalloy'.
    """
    return Material.fromDB("Py", **params)

  @staticmethod
  def Fe(**params):
    """
    Return a material object for the material 'Iron'.
    """
    logger.warn("Warning: The Fe material parameters are not verified!")
    return Material.fromDB("Fe", **params)

  @staticmethod
  def Ni(**params):
    """
    Return a material object for the material 'Nickel'.
    """
    logger.warn("Warning: The Ni material parameters are not verified!")
    return Material.fromDB("Ni", **params)

  @staticmethod
  def Co(**params):
    """
    Return a material object for the material 'Cobalt'.
    """
    logger.warn("Warning: The Co material parameters are not verified!")
    return Material.fromDB("Co", **params)

  @staticmethod
  def Au(**params):
    """
    Return a material object for the non-magnetic material 'Gold'.
    """
    return Material.fromDB("Au", **params)

  # Materials database #

  # From OOMMF source code: ###################################################
  #                                                                           #
  # PLEASE NOTE: The following values should *not* be taken as standard       #
  # reference values for these materials.  These values are only approximate. #
  # They are included here as examples for users who wish to supply their     #
  # own material types with symbolic names.                                   #
  #                  M_s        A         K      anisotropy type              #
  # Permalloy       860E3    13E-12       0      uniaxial                     #
  # Iron           1700E3    21E-12      48E3    cubic                        #
  # Nickel          490E3     9E-12    -5.7E3    cubic                        #
  # Cobalt(hcp)    1400E3    30E-12     520E3    uniaxial                     #
  #############################################################################

  __db = {
    # Permalloy
    'Py': {
        'id': 'Py', 
        # For LandauLifshitzGilbert module
        'Ms':  8e5, 
        'alpha': 0.01, 
        # ExchangeField
        'A': 13e-12,        # Exchange stiffness constant: J/m
        # CurrentPath
        'sigma': 0.5e7,     # A/(V*m) = 1/(Ohm*m)
        'amr': 0.1,
        # SpinTorque
        'P': 0.5,           # Spin polarization (1)
        'xi': 0.02,
    },

    # Iron: (Not verified!)
    'Fe': {
        'id': 'Fe',
        'Ms': 17e5,
        'alpha': 0.01,
        'A': 21e-12,
        'axis1': (1.0, 0.0, 0.0),
        "k_uniaxial": 48e3,
    },

    # Nickel: (Not verified!)
    'Ni': {
        'id': 'Ni',
        'Ms': 4.9e5,
        'alpha': 0.01,
        'A':  9e-12,
        'axis1': (1.0, 0.0, 0.0),
        "k_uniaxial": -5.7e3,
    },

    # Cobalt (source: OOMMF)
    'Co': {
        'id': 'Co',
        'Ms': 14e6,
        'alpha': 0.5,
        'A': 30e-12,
        'axis1': (1.0, 0.0, 0.0),
        'k_uniaxial': 520e3,
    },

    # Gold
    'Au': {
        'id': 'Au',
        'Ms': 0,
        'alpha': 0,
        'A': 0,
        'sigma': 4.55e7, # from wikipedia:en
        'amr': 0
    }
  }

  @staticmethod
  def __getDBParameters(id):
    return Material.__db[id]
