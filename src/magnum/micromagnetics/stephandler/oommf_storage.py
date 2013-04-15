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

from .storage_stephandler import StorageStepHandler

from magnum.micromagnetics.io import writeOMF, OMF_FORMAT_ASCII

class OOMMFStorage(StorageStepHandler):
    def __init__(self, output_dir, field_id_or_ids, omf_format = OMF_FORMAT_ASCII):
        super(OOMMFStorage, self).__init__(output_dir)

        if hasattr(field_id_or_ids, "__iter__"):
            field_ids = list(field_id_or_ids)
        else:
            field_ids = [field_id_or_ids]
        if not all(isinstance(x, str) for x in field_ids):
            raise ValueError("OOMMFStorage: 'field_id' parameter must be a either a string or a collection of strings.")

        self.__omf_format = omf_format

        def make_file_fn(field_id):
            pattern = "%s-%%07i" % field_id
            if field_id[0] == "H":
                pattern += ".ohf"
            else:
                pattern += ".omf"
            return lambda state: pattern % state.step
        for field_id in field_ids:
            self.addVariable(field_id, make_file_fn(field_id))

        self.addComment("time", lambda state: state.t)
        self.addComment("stepsize", lambda state: state.h)
        self.addComment("step", lambda state: state.step)

    def store(self, id, path, field, comments):
        writeOMF(path, field, ["%s = %s" % (key, val) for key, val in comments], self.__omf_format)
