MicroMagNum project homepage: http://micromagnum-tis.informatik.uni-hamburg.de

MicroMagnum
Fast Physical Simulator for Computations on CPU and Graphics Processing Unit (GPU)

MicroMagnum is a fast easy-to-use simulator that runs on CPUs as well as on GPUs using the CUDA platform. It combines the speed and flexibility of C++ together with the usability of Python.

MicroMagnum has a robust and highly modular architecture. This enables its easy extension by further physical modules.

--------------

Some tweaks can be enabled by setting environment variables 
to e.g. "1" (or "0" for that matter):

  MAGNUM_DEMAG_GARBAGE                
    Don't calculate demag tensor (and thus produce invalid results). 
    Useful for benchmarking only.

  MAGNUM_DEMAG_NO_INFINITY_CORRECTION 
    Disable demag tensor infinity correction in case of periodic boundary 
    conditions.

  MAGNUM_OMF_NOSCALE    
    Always use "valuemultiplier=1" for writing .omf/.ohf files.

Example 1:
  export MAGNUM_OMF_NOSCALE=1 && ./myscript.py

Example 2: (In Python script)
  from magnum import *
  import sys; sys.environ["MAGNUM_OMF_NOSCALE"] = "1"
  # do stuff
