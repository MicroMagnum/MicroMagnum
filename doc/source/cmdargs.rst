Command line arguments
======================

Any script that loads MicroMagnum using the 'import magnum' command is subject to these command line arguments:

.. code-block:: text

   Usage: <your script>.py [options]
   
   Options:
     --version             show program's version number and exit
     -h, --help            show this help message and exit
   
     Hardware options:
       Options that control which hardware is used.
   
       -g GPU_ID           enable GPU processing (using 32-bit accuracy) on cuda
                           device GPU_ID. The simulator will fall back to CPU
                           mode if it was not compiled with CUDA support or when
                           no CUDA capable graphics cards were detected.
       -G GPU_ID           enable GPU processing (using 64-bit accuracy) on cuda
                           device GPU_ID. TODO: Describe fallback behaviour.
       -t NUM_THREADS, --threads=NUM_THREADS
                           enable CPU multithreading with NUM_THREADS (1..64)
                           threads. This parameter instructs the fftw library to
                           use NUM_THREADS threads for computing FFTs.
   
     Logging options:
       Options related to logging and benchmarking.
   
       -l LEVEL, --loglevel=LEVEL
                           set log level (0:Debug, 1:Info, 2:Warn, 4:Error,
                           5:Critical), default is Debug (0).
       --prof              Log profiling info at program exit.
   
     Parameter sweep options:
       These options have only an effect when the simulation script uses a
       Controller object to sweep through a parameter range.
   
       -p RANGE, --param-range=RANGE
                           set parameter range to run, e.g. --prange=0,64.
       --print-num-params  print number of sweep parameters to stdout and exit.
       --print-all-params  print all sweep parameters to stdout and exit.
   
     Miscellanous options:
       --on_io_error=MODE  Specifies what to do when an i/o error occurs when
                           writing an .omf/.vtk file. 0: Abort (default), 1:
                           Retry a few times, then abort, 2: Retry a few times,
                           then pause and ask for user intervention

Tweaks
------

Some tweaks can be definging certain environment variables before
running a script. These are mostly useful for debugging and benchmarking the 
simulator.

* MAGNUM_DEMAG_GARBAGE

    Don't calculate demag tensor (and thus produce invalid results). 
    Useful for benchmarking only.

* MAGNUM_DEMAG_NO_INFINITY_CORRECTION

    Disable demag tensor infinity correction in case of periodic boundary 
    conditions.

* MAGNUM_OMF_NOSCALE

    Always use "valuemultiplier=1" for writing .omf/.ohf files.

Example 1:

.. code-block:: bash

   export MAGNUM_OMF_NOSCALE=1 && ./myscript.py

Example 2: (Inside Python script)

.. code-block:: python

   import sys; sys.environ["MAGNUM_OMF_NOSCALE"] = "1"
   from magnum import *
   # do stuff

