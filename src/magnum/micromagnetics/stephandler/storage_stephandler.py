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

from magnum.solver import StepHandler

import os
import collections

class StorageStepHandler(StepHandler):

    class StorageId(object):
        def __init__(self, var_id, file_fn, field_fn):
            self.__id = var_id
            self.__file_fn = file_fn
            self.__field_fn = field_fn

        def getFileNameForState(self, state):
            return os.path.normpath(self.__file_fn(state))

        def getFieldForState(self, state):
            return self.__field_fn(state)

    def __init__(self, output_dir):
        super(StorageStepHandler, self).__init__()
        self.__ids = {} # maps var_id to StorageId instance
        self.__comments = []
        self.__output_dir = os.path.normpath(output_dir)
        if not os.path.isdir(self.__output_dir): os.makedirs(self.__output_dir)

    def addComment(self, name, fn):
        #if not isinstance(name, str) or not callable(fn): #2to3
        if not isinstance(name, str) or not isinstance(fn, collections.Callable):
            raise TypeError("StorageStepHandler.addComment: 'name' must be a string and 'fn' must be a function")
        self.__comments.append((name, fn))

    def getCommentsForState(self, state):
        return [(name, str(fn(state))) for name, fn in self.__comments]

    def getOutputDirectory(self):
        return self.__output_dir

    def addVariable(self, var_id, file_fn, field_fn = None):
        if not field_fn: field_fn = lambda state: getattr(state, var_id)
        self.__ids[var_id] = StorageStepHandler.StorageId(var_id, file_fn, field_fn)

    def handle(self, state):
        comments = self.getCommentsForState(state)
        for id, sid in self.__ids.items():
            path  = os.path.normpath(self.__output_dir + "/" + sid.getFileNameForState(state))
            field = sid.getFieldForState(state)
            self.store(id, path, field, comments)

    def store(self, id, path, field, comments):
        raise NotImplementedError("StorageStepHandler.store is purely virtual.")
