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

import os
import collections

from magnum.solver import StepHandler


class StorageStepHandler(StepHandler):

    class StorageId(object):
        def __init__(self, var_id, file_fn, field_fn):
            self.var_id = var_id
            self.file_fn = file_fn
            self.field_fn = field_fn

        def getFileNameForState(self, state):
            return os.path.normpath(self.file_fn(state))

        def getFieldForState(self, state):
            return self.field_fn(state)

    def __init__(self, output_dir):
        super(StorageStepHandler, self).__init__()
        self.ids = {}  # maps var_id to StorageId instance
        self.comments = []
        self.output_dir = os.path.normpath(output_dir)
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)

    def addComment(self, name, fn):
        if not isinstance(name, str) or not isinstance(fn, collections.Callable):
            raise TypeError("StorageStepHandler.addComment: 'name' must be a string and 'fn' must be a function")
        self.comments.append((name, fn))

    def getCommentsForState(self, state):
        return [(name, str(fn(state))) for name, fn in self.comments]

    def getOutputDirectory(self):
        return self.output_dir

    def addVariable(self, var_id, file_fn, field_fn=None):
        if not field_fn:
            field_fn = lambda state: getattr(state, var_id)
        self.ids[var_id] = StorageStepHandler.StorageId(var_id, file_fn, field_fn)

    def handle(self, state):
        comments = self.getCommentsForState(state)
        for id, sid in self.ids.items():
            path  = os.path.normpath(self.output_dir + "/" + sid.getFileNameForState(state))
            field = sid.getFieldForState(state)
            self.store(id, path, field, comments)

    def store(self, id, path, field, comments):
        raise NotImplementedError("StorageStepHandler.store is purely virtual.")
