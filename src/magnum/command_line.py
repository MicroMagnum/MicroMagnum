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

import optparse
import re

import magnum.logger as logger


def parse(argv, version):
    parser = optparse.OptionParser(version=version)

    hw_group = optparse.OptionGroup(
        parser,
        "Hardware options",
        "Options that control which hardware is used."
    )

    hw_group.add_option(
        "-g",
        type="string",
        help="Enable GPU processing (using 32-bit accuracy) on CUDA device with"
             "device-id GPU_ID. The simulator will fall back to CPU mode if it "
             "was not compiled with CUDA support or when no CUDA capable "
             "graphics cards were detected.",
        metavar="GPU_ID",
        dest="gpu32",
        default=None,
    )

    hw_group.add_option(
        "-G",
        type="string",
        help="Enable GPU processing (using 64-bit accuracy for all terms except "
             "the demagnetization field computation) on a CUDA device with "
             "device-id GPU_ID. TODO: Describe fallback behaviour.",
        metavar="GPU_ID",
        dest="gpu64",
        default=None,
    )

    hw_group.add_option(
        "-t",
        help="Enable CPU multithreading with NUM_THREADS (1..64) threads. "
             "This parameter instructs the fftw library to use NUM_THREADS "
             "threads for computing fast Fourier transforms. (Don't expect "
             "too much from this option.)",
        metavar="NUM_THREADS",
        dest="num_fftw_threads",
        type="int",
        default=1,
    )

    log_group = optparse.OptionGroup(
        parser,
        "Logging options",
        "Options related to logging and benchmarking."
    )

    ll = ("%s:%s" % (i, dsc[1]) for i, dsc in enumerate(logger.LOGLEVELS))

    log_group.add_option(
        "-l",
        type="int",
        help="Supress all log output below the level given by LEVEL. "
             "Valid values are (%s), default is 1." % ", ".join(ll),
        metavar="LEVEL",
        dest="loglevel",
        default=1,
    )

    log_group.add_option(
        "--prof",
        help="Log profiling info at program exit.",
        dest="profiling_enabled",
        action="store_true",
        default=False,
    )

    def p_callback(option, opt_str, value, parser, *args, **kwargs):

        match0 = re.match("^(\d+),(\d+)$", value)  # -p <from>,<to>
        match1 = re.match("^(\d+)$", value)        # -p <idx>

        if match0:
            p0 = int(match0.group(1))
            p1 = int(match0.group(2))
        elif match1:
            p0 = int(match1.group(1))
            p1 = p0 + 1
        else:
            raise optparse.OptionValueError(
                "Invalid -p argument: "
                "Valid forms: '-p <idx>' or '-p <from>,<to>'."
            )

        if p0 < 0 or p1 < 0:
            raise optparse.OptionValueError(
                "Invalid -p argument: "
                "Indices must be positive."
            )

        if not p0 < p1:
            raise optparse.OptionValueError(
                "Invalid -p argument: "
                "First value must be smaller than first value."
            )

        setattr(parser.values, "prange", range(p0, p1))

    ctrl_group = optparse.OptionGroup(
        parser,
        "Parameter sweep options",
        "These options have only an effect when the simulation script uses "
        "a Controller object to sweep through a parameter range."
    )

    ctrl_group.add_option(
        "-p",
        type="string",
        help="Select parameter sets to run when a Controller object is used. "
             "Each parameter set has an index. "
             "Valid forms: "
             "(a) '-p <from>,<to>' to use all sets in the range [from..(to-1)]; "
             "(b) '-p <idx>' to use the set 'idx' only.",
        metavar="RANGE",
        action="callback",
        callback=p_callback,
        default=None,
    )

    ctrl_group.add_option(
        "--print-num-params",
        help="Print the number of parameters set to stdout and exit.",
        action="store_true",
        dest="print_num_params",
        default=False,
    )

    ctrl_group.add_option(
        "--print-all-params",
        action="store_true",
        help="Print all parameter sets to stdout and exit.",
        dest="print_all_params",
        default=False,
    )

    misc_group = optparse.OptionGroup(parser, "Miscellanous options")
    misc_group.add_option(
        "--on-io-error",
        type="int",
        help="Specify what to do when an i/o error occurs when writing an "
             ".omf or .vtk file: (0:abort, 1:retry a few times, then abort, "
             "2:retry a few times, then pause and ask for user intervention), "
             "default is 0.",
        metavar="MODE",
        dest="on_io_error",
        default=0
    )

    parser.add_option_group(hw_group)
    parser.add_option_group(log_group)
    parser.add_option_group(ctrl_group)
    parser.add_option_group(misc_group)
    return parser.parse_args(argv)
