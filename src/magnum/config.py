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
import socket  # gethostname()

import magnum.logger as logger
import magnum.magneto as magneto
import magnum.tools as tools
import magnum.command_line as command_line
import magnum.nvidia_smi as nvidia_smi


class Configuration(object):

    def initialize(self, argv):
        magneto.initialize(self.cache_directory)

        self.options, rest_args = command_line.parse(argv, self.version)
        self.processCommandLine(self.options)

    def deinitialize(self):
        if self.isProfilingEnabled():
            magneto.printProfilingReport()

        magneto.deinitialize(self.cache_directory)

    @property
    def global_cache_directory(self):
        global_cache_dir = os.path.expanduser("~/.cache/magnum")
        tools.makedirs(global_cache_dir)
        return global_cache_dir

    @property
    def cache_directory(self):
        cache_dir = self.global_cache_directory + "/" + socket.gethostname()
        tools.makedirs(cache_dir)
        return cache_dir

    @property
    def banner(self):
        return [
            "---------------------------------------------------------------------",
            self.version,
            "Copyright (C) 2012, 2013 by the MicroMagnum team.",
            "This program comes with ABSOLUTELY NO WARRANTY.",
            "This is free software, and you are welcome to redistribute it under",
            "certain conditions; see the file COPYING in the distribution package.",
            "---------------------------------------------------------------------",
        ]

    @property
    def version(self):
        import magnum
        return "MicroMagnum " + magnum.__version__

    def enableCuda(self, mode, device=-1):
        return magneto.enableCuda(mode, device)

    def isCudaEnabled(self):
        return magneto.isCudaEnabled()

    def isCuda64Enabled(self):
        return magneto.isCuda64Enabled()

    def haveCudaSupport(self):
        return magneto.haveCudaSupport()

    def enableProfiling(self, yes=True):
        magneto.enableProfiling(yes)

    def isProfilingEnabled(self):
        return magneto.isProfilingEnabled()

    def haveFFTWThreads(self):
        return magneto.haveFFTWThreads()

    def setFFTWThreads(self, num_threads):
        magneto.setFFTWThreads(num_threads)

    def processCommandLine(self, options):
        # Not processed here:
        #   -p, --print-num-params, --print-all-params
        #   --on-io-error

        # Loglevel: -l
        logger.logger.setLevel(logger.LOGLEVELS[options.loglevel][0])

        for line in self.banner:
            logger.info(line)

        # Profiling: --prof
        if options.profiling_enabled:
            self.enableProfiling(True)
            logger.info("Profiling enabled")

        # Number of fftw threads: -t
        if options.num_fftw_threads != 1:
            self.setFFTWThreads(options.num_fftw_threads)

        # GPU enable: -g, -G
        if options.gpu32 and options.gpu64:
            logger.warn("Ignoring -g because -G was given")

        def determine_gpu_id(arg):
            # 'auto'
            if arg == "auto":
                try:
                    gpus = nvidia_smi.run()
                    available = nvidia_smi.available(gpus)
                    if len(available == 0):
                        logger.warn("Could not find any available GPUs (out of %s detected GPUs)." % len(gpus))
                        return -1
                    return available[0]
                except:
                    return -1
            else:
                # id specified by user
                return int(arg)

        if options.gpu64:
            cuda_mode, cuda_dev = magneto.CUDA_64, determine_gpu_id(options.gpu64)
        elif options.gpu32:
            cuda_mode, cuda_dev = magneto.CUDA_32, determine_gpu_id(options.gpu32)
        else:
            cuda_mode, cuda_dev = magneto.CUDA_DISABLED, -1

        if cuda_mode != magneto.CUDA_DISABLED and not self.haveCudaSupport():
            logger.error("Can't enable GPU computation (-g or -G flags): CUDA libraries could not be loaded.")
            logger.error("-> Falling back to CPU.")

        self.enableCuda(cuda_mode, cuda_dev)

        cuda_support = "yes" if self.haveCudaSupport() else "no"
        logger.info("CUDA GPU support: %s", cuda_support)
        logger.info("Running on host %s", socket.gethostname())


# Main configuration object, at this point still uninitialized.
# - initialize is called at the end of __init__.py
# - deinitialize is called at program exit, see __init__.py
cfg = Configuration()
initialize = cfg.initialize
deinitialize = cfg.deinitialize
