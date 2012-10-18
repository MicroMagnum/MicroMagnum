Controllers
===========

.. currentmodule:: magnum

.. autofunction:: Controller

   This function creates a controller object that is responsible for starting multiple simulations e.g. for parameter sweeps.
   Depending on how the simulation script is started, an instance of one of the following controller classes is created:
  
   - LocalController
   - (more to follow)

   If the script is started locally, a LocalController instance is returned.

   Example on how to use controllers:
   
   .. code-block:: python

      mesh = RectangularMesh(...)
      world = World(mesh, ...)

      def run(n):
        print "Running with parameter", n
        solver = Solver(world)
        # etc...
        solver.solve(...)

      controller = Controller(run, [100, 200, 300, 400])
      controller.start()

   This will run 4 simulations, printing:

   .. code-block:: text

      Running with parameter 100
      Running with parameter 200
      Running with parameter 300
      Running with parameter 400

.. class:: LocalController

   .. automethod:: start(x,y,z)
