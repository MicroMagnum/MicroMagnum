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

from magnum.mesh import Field, VectorField
from magnum.logger import logger

import numbers

def assign(obj, val, mask=None):
    global _typemap_assign
    try:
        return _typemap_assign[type(obj)](obj, val, mask)
    except KeyError:
        # Don't know how to process 'mask' in default behavour:
        if mask: raise ValueError("assign: Don't know how to apply mask by default")
        # Defaut behavour: Reference semantics.
        return val

def assign_Field(field, val, mask):
    mesh = field.mesh

    # initialize from lambda?
    from_lambda = hasattr(val, "__call__")
    # initialize from other Field???
    from_field = isinstance(val, Field)
    # initialize from a numerical value?
    from_value = not from_lambda and not from_field and isinstance(val, numbers.Number)
    # fill entire field?
    fill_everywhere = (mask is None) or (len(mask) == mesh.total_nodes)

    # Do the initialization (modify field)
    if from_value:
        if fill_everywhere:
            # Fill complete field with a single value
            field.fill(float(val))
        else:
            # Masked fill with a single value
            for c in mask: field.set(c, val)

    elif from_field:
        src_field = val

        # Automatic field interpolation if meshes do not match.
        # XXX Is this a good idea?
        if not field.mesh.isCompatible(src_field.mesh):
            src_field = src_field.interpolate(field.mesh)
            logger.warn("Field initialized by linear interpolation of other field (their meshes were different sizes!)")

        if fill_everywhere:
            # Fill field by copying values from init field
            field.assign(src_field)
        else:
            # Fill some parts of the field using values from field
            for c in mask: field.set(c, src_field.get(c))

    elif from_lambda:
        fn = val
        # Fill some parts of the field using an initialization function
        for c in mask or range(mesh.total_nodes):
            field.set(c, fn(field, mesh.getPosition(c)))

    else:
        raise ValueError("Don't know how to assign to field using the expression %s" % val)

    return field

def assign_VectorField(field, val, mask):
    mesh = field.mesh

    # initialize from lambda?
    from_lambda = hasattr(val, "__call__")
    # initialize from other VectorField ???
    from_field = isinstance(val, VectorField)
    # initialize from a numerical value?
    from_value = not from_lambda and not from_field and isinstance(val, tuple) and len(val) == 3
    # fill entire field?
    fill_everywhere = (mask is None) or (len(mask) == mesh.total_nodes)

    # Do the initialization (modify field)
    if from_value:
        if fill_everywhere:
            # Fill complete field with a single value
            field.fill(val)
        else:
            # Masked fill with a single value
            for c in mask: field.set(c, val)

    elif from_field:
        src_field = val

        # Automatic field interpolation if meshes do not match.
        # XXX Is this a good idea?
        if not field.mesh.isCompatible(src_field.mesh):
            src_field = src_field.interpolate(field.mesh)
            logger.warn("Field initialized by linear interpolation of other field (their meshes were different sizes!)")

        if fill_everywhere:
            # Fill field by copying values from init field
            field.assign(src_field)
        else:
            # Fill some parts of the field using values from field
            for c in mask: field.set(c, src_field.get(c))

    elif from_lambda:
        fn = val
        # Fill some parts of the field using an initialization function
        for c in mask or range(mesh.total_nodes):
            field.set(c, fn(field, mesh.getPosition(c)))

    else:
        raise ValueError("Don't know how to assign to vector field using the expression %s" % val)

    return field

_typemap_assign = {
    Field: assign_Field,
    VectorField: assign_VectorField
}
