===========================
 Input and output routines
===========================

Read and write vector fields
----------------------------

Supported file formats: 
- .omf (read & write)

  OOMMF field files.

- .png (read only)
 
  (Also any other graphics file format that is supported by the Python imaging library.)
  
  Graphics files can be used to load vector fields.  

- .vtr (write only)

  VTK rectangular mesh vector fields.

Logging functions
-----------------

Example:

.. code-block:: python
  
  from magnum.logger import logger

  # From most to least severe:
  logger.critical("Red alert!")
  logger.error("This is an error")
  logger.warn("This is a warning")
  logger.info("This is an informational message")
  logger.debug("Debug message")

  # Formated messages:
  x = 12
  y = 15
  logger.info("x is %s and y is %s", x, y)

