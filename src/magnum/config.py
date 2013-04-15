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

import os, re, sys, socket, atexit, logging
from optparse import OptionParser, OptionValueError, OptionGroup

from magnum.logger import logger
import magnum.magneto as magneto

class MagnumConfig(object):

    def initialize(self, argv=None):
        # Initialize magneto
        magneto.initialize(self.getCacheDirectory())
        self.parseCommandLine(argv or sys.argv)

        # Register cleanup function
        atexit.register(MagnumConfig.cleanupBeforeExit, self)

        logger.info("CUDA GPU support: %s", "yes" if self.haveCudaSupport() else "no")

    def cleanupBeforeExit(self):
        if self.isProfilingEnabled(): magneto.printProfilingReport()
        magneto.deinitialize(self.getCacheDirectory())

    @staticmethod
    def __ensureDirectoryExists(path):
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                # Sometimes many parallel MicroMagNum are started at the same time.
                # Another process might have created the directory already, so that the os.makedirs call above fails.
                # In this case, we can savely ignore the exception if the path now exists.
                if not os.path.exists(path): raise
        return path

    def getGlobalCacheDirectory(self):
        global_dir = os.path.expanduser("~/.cache/magnum")
        self.__ensureDirectoryExists(global_dir)
        return global_dir

    def getCacheDirectory(self):
        cache_dir = os.path.expanduser("~/.cache/magnum") + "/" + socket.gethostname()
        self.__ensureDirectoryExists(cache_dir)
        return cache_dir

    global_cache_directory = property(getGlobalCacheDirectory)
    cache_directory = property(getCacheDirectory)

    CUDA_DISABLED = magneto.CUDA_DISABLED
    CUDA_32 = magneto.CUDA_32
    CUDA_64 = magneto.CUDA_64

    def enableCuda(self, mode, device=-1):
        return magneto.enableCuda(mode, device)

    def isCudaEnabled(self):
        return magneto.isCudaEnabled()

    def isCuda64Enabled(self):
        return magneto.isCuda64Enabled()

    def haveCudaSupport(self):
        return magneto.haveCudaSupport()

    def enableProfiling(self, yes = True):
        magneto.enableProfiling(yes)
        if magneto.isProfilingEnabled():
            logger.info("Profiling enabled")
        else:
            logger.info("Profiling disabled")

    def isProfilingEnabled(self):
        return magneto.isProfilingEnabled()

    def haveFFTWThreads(self):
        return magneto.haveFFTWThreads()

    def setFFTWThreads(self, num_threads):
        magneto.setFFTWThreads(num_threads)

    def parseCommandLine(self, argv):
        import magnum

        parser = OptionParser(version="MicroMagnum " + magnum.__version__)

        hw_group = OptionGroup(parser,
          "Hardware options",
          "Options that control which hardware is used.",
        )
        hw_group.add_option("-g",
          type    = "string",
          help    = "enable GPU processing (using 32-bit accuracy) on cuda device GPU_ID. The simulator will fall back to CPU mode if it was not compiled with CUDA support or when no CUDA capable graphics cards were detected.",
          metavar = "GPU_ID",
          dest    = "gpu32",
          default = None
        )
        hw_group.add_option("-G",
          type    = "string",
          help    = "enable GPU processing (using 64-bit accuracy) on cuda device GPU_ID. TODO: Describe fallback behaviour.",
          metavar = "GPU_ID",
          dest    = "gpu64",
          default = None
        )
        hw_group.add_option("-t", "--threads",
          help    = "enable CPU multithreading with NUM_THREADS (1..64) threads. This parameter instructs the fftw library to use NUM_THREADS threads for computing FFTs.",
          metavar = "NUM_THREADS",
          dest    = "num_fftw_threads",
          type    = "int",
          default = 1
        )
        parser.add_option_group(hw_group)

        log_group = OptionGroup(parser,
          "Logging options",
          "Options related to logging and benchmarking."
        )
        log_group.add_option("--loglevel", "-l",
          type    = "int",
          help    = "set log level (0:Debug, 1:Info, 2:Warn, 4:Error, 5:Critical), default is Debug (0).",
          metavar = "LEVEL",
          dest    = "loglevel",
          default = 1
        )
        log_group.add_option("--prof",
          help    = "Log profiling info at program exit.",
          dest    = "profiling_enabled",
          action  = "store_true",
          default = False
        )
        parser.add_option_group(log_group)

        def prange_callback(option, opt_str, value, parser, *args, **kwargs):
            # e.g.: ./script.py --prange=0,64   -> xrange(0,64) object
            match = re.match("^(\d+),(\d+)$", value)
            if match:
                p0, p1 = int(match.group(1)), int(match.group(2))
            else:
                match = re.match("^(\d+)$", value)
                if match:
                    p0 = int(match.group(1))
                    p1 = p0 + 1
                else:
                    raise OptionValueError("Invalid --param-range / -p format.")
            if not p0 < p1:
                raise OptionValueError("Invalid --param-range / -p specified: second value must be greater than first value")
            setattr(parser.values, "prange", range(p0, p1))

        ctrl_group = OptionGroup(parser,
          "Parameter sweep options",
          "These options have only an effect when the simulation script uses a Controller object to sweep through a parameter range."
        )
        ctrl_group.add_option("--param-range", "-p",
          type    = "string",
          help    = "select parameter set to run, e.g. --param-range=0,64 to run sets 0 to 63.",
          metavar = "RANGE",
          action  = "callback",
          callback= prange_callback,
          default = None
        )
        ctrl_group.add_option("--print-num-params",
          help    = "print number of sweep parameters to stdout and exit.",
          action  = "store_true",
          dest    = "print_num_params_requested",
          default = False
        )
        ctrl_group.add_option("--print-all-params",
          action  = "store_true",
          help    = "print all sweep parameters to stdout and exit.",
          dest    = "print_all_params_requested",
          default = False,
        )
        parser.add_option_group(ctrl_group)

        misc_group = OptionGroup(parser,
          "Miscellanous options"
        )
        misc_group.add_option("--on_io_error",
          type = "int",
          help = "Specifies what to do when an i/o error occurs when writing an .omf/.vtk file. 0: Abort (default), 1: Retry a few times, then abort, 2: Retry a few times, then pause and ask for user intervention",
          metavar = "MODE",
          dest = "on_io_error",
          default = 0
        )
        parser.add_option_group(misc_group)

        options, rest_args = parser.parse_args(argv)
        self.options = options

        ### Process the options ###

        # --loglevel, -l
        ll_map = [logging.DEBUG, logging.INFO, logging.WARN, logging.ERROR, logging.CRITICAL]
        logger.setLevel(ll_map[options.loglevel])

        logger.info("----------------------------------------------------------------------")
        logger.info("MicroMagnum %s" % magnum.__version__)
        logger.info("Copyright (C) 2012 by the MicroMagnum team.")
        logger.info("This program comes with ABSOLUTELY NO WARRANTY.")
        logger.info("This is free software, and you are welcome to redistribute it under")
        logger.info("certain conditions; see the file COPYING in the distribution package.")
        logger.info("----------------------------------------------------------------------")

        # -g, -G
        if options.gpu32 and options.gpu64: logger.warn("Ignoring -g argument because -G was specified")

        def parse_gpu_id(arg): return -1 if arg == "auto" else int(arg)
        if   options.gpu64: cuda_mode, cuda_dev = MagnumConfig.CUDA_64, parse_gpu_id(options.gpu64)
        elif options.gpu32: cuda_mode, cuda_dev = MagnumConfig.CUDA_32, parse_gpu_id(options.gpu32)
        else:               cuda_mode, cuda_dev = MagnumConfig.CUDA_DISABLED, -1

        self.enableCuda(cuda_mode, cuda_dev)

        # --prof
        if options.profiling_enabled:
            self.enableProfiling(True)

        # enable fftw threads...
        self.setFFTWThreads(options.num_fftw_threads)

# Main configuration object, at this point still uninitialized.
# cfg.initialize() is called at the end of __init__.py
cfg = MagnumConfig()
