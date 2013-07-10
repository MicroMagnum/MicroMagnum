============
Installation
============

In this section the following tasks are described:

- Installing required libraries
- Building MicroMagnum from source
- Testing the MicroMagnum installation

Required libraries
------------------

Required runtime libraries:

+---------------------------+---------------+-------------+-----------------------------------+---------------------------------------------------+-----------------------+
| Library                   | Should work   | Recommended | Remarks                           | URL                                               | Ubuntu/Debian package |
|                           | with version  | version     |                                   |                                                   |                       |
+===========================+===============+=============+===================================+===================================================+=======================+
| Python developer package  |      2.4      | 2.6,2.7,3.2 | Python 3.x works                  | http://www.python.org                             | python-dev            |
+---------------------------+---------------+-------------+-----------------------------------+---------------------------------------------------+-----------------------+
| FFTW                      |     3.0.0     |    3.3.2+   | Get the highest version           | http://www.fftw.org                               | libfftw3-dev          |
+---------------------------+---------------+-------------+-----------------------------------+---------------------------------------------------+-----------------------+
| CUDA                      | CUDA 3        |   highest   | Tested with CUDA 4                | http://www.nvidia.com                             | (n/a)                 |
| (if GPU is enabled)       |               |             |                                   |                                                   |                       |
+---------------------------+---------------+-------------+-----------------------------------+---------------------------------------------------+-----------------------+
| Numpy                     |     1.3.0     |    1.3.0+   | Needed to convert scalar/vector   | http://www.numpy.org                              | python-numpy          |
| (optional)                |               |             | field objects to numpy arrays     |                                                   |                       |
+---------------------------+---------------+-------------+-----------------------------------+---------------------------------------------------+-----------------------+
| Python Imaging Library    |   any recent  | any recent  | Needed for the ImageShapeCreator  | http://www.pythonware.com/products/pil/           | python-imaging        |
| (optional)                |               |             | class (optional)                  |                                                   |                       |
+---------------------------+---------------+-------------+-----------------------------------+---------------------------------------------------+-----------------------+
| Sundials CVode            | sundials-2.5.0| cvode-2.7.0 | Needed for using implicit CVode   | http://computation.llnl.gov/casc/sundials         | (n/a)                 |
| (optional)                | cvode-2.7.0   |             | evolver (optional)                |                                                   |                       |
+---------------------------+---------------+-------------+-----------------------------------+---------------------------------------------------+-----------------------+

Ubuntu/Debian apt-get command:

.. code-block:: bash

   sudo apt-get install python-dev python-numpy python-imaging libfftw3-dev

Additional software that is required to build MicroMagnum from source:

+--------------------+---------------+-------------+--------------------------------+------------------------------+-----------------------+
| Software           | "should work" | Recommended | Remarks                        | URL                          | Ubuntu/Debian package |
+====================+===============+=============+================================+==============================+=======================+
| g++ build chain    |      any      |    newest   |                                | http://gcc.gnu.org           | g++                   |
+--------------------+---------------+-------------+--------------------------------+------------------------------+-----------------------+
| cmake              |     2.8.8     |    2.8.8+   | Build system                   | http://www.cmake.org         | cmake                 |
+--------------------+---------------+-------------+--------------------------------+------------------------------+-----------------------+
| swig               |      1.3      |    2.0.1+   | Python wrapper generator       | http://www.swig.org          | swig1.3, swig2.0      |
+--------------------+---------------+-------------+--------------------------------+------------------------------+-----------------------+
| Sphinx             | any recent    |    0.6.6+   | Python documentation generator | http://sphinx.pocoo.org      | python-sphinx         |
+--------------------+---------------+-------------+--------------------------------+------------------------------+-----------------------+
| bzr                | any recent    | any recent  | Bazaar version control system  | http://bazaar-vcs.org/       | bzr                   |
+--------------------+---------------+-------------+--------------------------------+------------------------------+-----------------------+

Ubuntu/Debian apt-get command:

.. code-block:: bash

   sudo apt-get install g++ cmake python-sphinx bzr
   sudo apt-get install swig2.0
   sudo apt-get install swig1.3 # if swig2.0 is not available

Building MicroMagnum from source
--------------------------------

Getting the source code: You may either get an source package from the download area or get the source from the official git repository.

.. code-block:: bash

   # either from Github 
   git clone git://github.com/MicroMagnum/MicroMagnum.git

   # or extract a source package..
   tar -xvf micromagnum.tar.gz

Now the MicroMagnum source code is located in the "micromagnum" subdirectory. To build MicroMagnum, enter:

.. code-block:: bash
  
   cd micromagnum                  # enter MicroMagnum base directory
   cd src/build                    # enter build directory
   cmake ..                        # to compile for CPU
   make                            # start the build process
   sudo make install               # installs MicroMagnum as python package

You can use the following parameters to customize your installation:

.. code-block:: bash

   cmake .. [parameters]
   -DENABLE_CUDA_32=on  # to compile for cuda 32 bit
   -DENABLE_CUDA_64=on  # to compile for cuda 64 bit
   -DUSE_PYTHON2=on     # force compile for python2
   -DUSE_PYTHON3=on     # force compile for python3
   -DUSE_CVODE=on       # compile with CVode

If you don't want to install MicroMagnum, then set PYTHONPATH to <micromagnum-basedir>/src

To check the installation, start the Python interactive shell by entering "python"
and type "import magnum". If everything went ok, you should see no errors like this:

.. code-block:: bash

   Python 2.7.1+ (r271:86832, Apr 11 2011, 18:13:53) 
   [GCC 4.5.2] on linux2
   Type "help", "copyright", "credits" or "license" for more information.
   >>> import magnum
   [ INFO] - MicroMagnum 0.2 Copyright (C) 2012 by the MicroMagnum team.
   [ INFO] - This program comes with ABSOLUTELY NO WARRANTY.
   [ INFO] - This is free software, and you are welcome to redistribute it
   [ INFO] - under certain conditions; see the file COPYING in the distribution package.
   >>> 

You can then enter the examples directory in the MicroMagnum base directory and try out the examples, like:

.. code-block:: bash

   cd examples/sp4
   ./sp4 -l0           # add -g0 to run on GPU 
                       #(this works only when GPU support was enabled at compile time)

Building with CVode evolver
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to use MicroMagnum with the implicit CVode evolver you can get it on the sundials download page.
http://computation.llnl.gov/casc/sundials/download/download.html

Or get the package directly:

.. code-block:: bash

   wget http://computation.llnl.gov/casc/sundials/download/code/cvode-2.7.0.tar.gz
   tar -xvf cvode-2.7.0.tar.gz

To build and install you can use:

.. code-block:: bash

  cd cvode-2.7.0          # change to the source directory
  ./configure --with-pic  # the PIC option is important to use it with MicroMagnum.
  make                    # build
  sudo make install       # and install cvode

Now you can build MicroMagnum and activate CVode with the toggle:

.. code-block:: bash

  cmake .. -DUSE_CVODE=on   # to enable CVode

If you do not have the permission to install globally, use these Cmake parameters:

.. code-block:: bash

  cmake .. -DUSE_CVODE=on -DCMAKE_INCLUDE_PATH=/INSTALLPATH/include -DCMAKE_LIBRARY_PATH=/INSTALLPATH/lib


FFTW download and building
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want a custom build of FFTW.

FFTW download, version 3.2.2 at http://www.fftw.org/fftw-3.3.2.tar.gz, see
http://www.fftw.org for newer versions such as the latest alpha version.

Suggested configure parameters:

.. code-block:: bash

   ./configure --with-pic --prefix=/home/gselke/fftw --enable-openmp --enable-sse2

