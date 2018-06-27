===============
Getting Started
===============

If you successfully followed the installation instructions you are now ready to start your first micromagnetic simulations with MicroMagnum. 
This tutorial will take you step by step through the simulation of the standard problem #4.

Create a Python Script
----------------------

MicroMagnum is a Python module. That means that all the functionality of MicroMagnum is accessed by writing Python scripts or by using the Python console. 
Open the text editor of your choice and create an empty text file named sp4.py. 
Every simulation script will start with the import of the MicroMagnum module.

.. code-block:: python

    from magnum import *

This line makes the classes and functions of MicroMagnum accessible in the script file.

The World
---------

Every micromagnetic simulation takes place in a virtual world. 
This world has a certain size and spatial discretization and it contains certain bodies. 
The size and discretization of the world is defined by a mesh object.

.. code-block:: python

    mesh = RectangularMesh((100, 25, 1), (5e-9, 5e-9, 3e-9))

For the standard problem #4 we use a rectangular mesh, which is initialized by a tuple of cell numbers (100, 25, 1) and a tuple of cells sizes (5e-9, 5e-9, 3e-9). 
Now we have to define at least one body of magnetic material we want to simulate.

.. code-block:: python

    body = Body("all", Material.Py(), Everywhere())

To create a new body we have to define a name ("all"), a material (Material.Py()) and a shape of the body (Everywhere()). 
In this case we created a permalloy body which fills the whole area defined by the mesh. Now we are ready to create our virtual world.

.. code-block:: python

    world = World(mesh, body)

A world is created by supplying a mesh and at least one body.

The solver
----------

Now that we have a virtual world with bodies in it we want to breathe some life into it. This is where the solver comes in.

.. code-block:: python

    solver = create_solver(world)

The first step to the solution of the standard problem #4 is the creation of an S-state in our sample. To create an S-state we start with a simple analytical expression for this configuration and use the solver to relax this configuration at the energetical minimum.

In order to initialize the magnetization of our sample we define a method that returns the values of our analytical expression.

.. code-block:: python

    from math import pi, cos, sin

    def state0(field, pos):
        u = abs(pi*(pos[0]/field.mesh.size[0]-0.5)) / 2.0
        return 8e5 * cos(u), 8e5 * sin(u), 0

Now we initialize the magnetization in our solver.

.. code-block:: python

    solver.state.M = state0

The only thing that is left to do is to tell the solver to relax the state which is done by a simple call to the solve function:

.. code-block:: python

    solver.solve(condition.Relaxed())

The solve method computes the time evolution of the system and stops when the termination condition is true. Since the relaxation of a system is a very common task, MicroMagnum defines a shorthand for this call.

.. code-block:: python

    solver.relax()

Applying a Field and Saving the Results
---------------------------------------

Now that we have produced an S-state we want to apply an external field as defined by the standard problem.

In order to use the relaxed state as initial state of a simulation we first save it.

.. code-block:: python

    s_state = solver.state.M

Then we create a new solver, set the S-state as initial state and add an external field to the solver

.. code-block:: python

    solver = create_solver(world, [StrayField, ExchangeField, ExternalField])
    solver.state.M = s_state
    solver.state.H_ext_offs = (-24.6e-3/MU0, +4.3e-3/MU0, 0.0)

Next we have to add step handlers to the solver that store our simulation results to the harddisk.

.. code-block:: python

    solver.addStepHandler(DataTableLog("sp4_1.odt"), condition.EveryNthStep(10))

and start the engines

.. code-block:: python

    solver.solve(condition.Time(1.0e-9))

Putting the Whole Thing Together
--------------------------------

And here is the resulting MicroMagnum script.

.. code-block:: python

    from magnum import *
    from math import pi, cos, sin

    mesh = RectangularMesh((100, 25, 1), (5e-9, 5e-9, 3e-9))
    body = Body("all", Material.Py(), Everywhere())
    world = World(mesh, body)

    # Relax to S-state
    def state0(field, pos):
        u = abs(pi*(pos[0]/field.mesh.size[0]-0.5)) / 2.0
        return 8e5 * cos(u), 8e5 * sin(u), 0

    solver = create_solver(world, [StrayField, ExchangeField], log=True, 
        do_precess=False, evolver="rkf45", eps_abs=1e-4, eps_rel=1e-2)
    solver.state.M = state0
    solver.relax()

    # Set external field and compute the time evolution
    s_state = solver.state.M

    solver = create_solver(world, [StrayField, ExchangeField, ExternalField])
    solver.state.M = s_state
    solver.state.H_ext_offs = (-24.6e-3/MU0, +4.3e-3/MU0, 0.0)
    solver.addStepHandler(DataTableLog("sp4_1.odt"), condition.EveryNthStep(10))
    solver.solve(condition.Time(1.0e-9))

You can run it by simply calling it with the python interpreter:

.. code-block:: bash

    $ python sp4.py

A more sophisticated version of the standard problem #4 can be found in the examples of Micromagnum.

Notes
-----

A Word on Materials and Shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MicroMagnum offers a set of predefined materials such as permalloy (Material.Py()) and cobalt (Material.Co()). 
However if the desired material is not in the database you can easily define your own material. 
MicroMagnum offers you two ways to define your own material.

Modify an existing Material:

.. code-block:: python

    py_with_high_damping = Material.Py(alpha = 0.5)

Create your own Material:

.. code-block:: python

    my_material = Material({
        'id': 'my_material',
        'Ms': 500e3,
        'alpha': 0.2,
        'A': 30e-12,
        'axis1': (0,0,1),
        'axis2': (0,1,0), 
        'k_uniaxial': 400e3,
        'k_cubic': 0.0})

If you omit the shape argument at the creation of a new body, MicroMagnum uses the Everywhere shape by default. 
This shape fills the whole space defined by the mesh. 
Beside this most simple shape MicroMagnum offers a variety of more sophisticated shapes including cuboids, cylinders and shapes defined by images.

Output / Stephandlers
~~~~~~~~~~~~~~~~~~~~~

Although MicroMagnum has a minimalistic GUI to show you the progress of your simulation. 
What you really want to do is to safe your simulation results on harddisk to analyse them with a post processing tool. 
MicroMagnum supports the following file formats:

- ODT
    A simple CSV format for the storage of scalar simulation data like the average magnetization over time
- OMF
    A file format for the storage of the magnetization configuration compatible to OOMMF
- VTK
    A file format for the storage of the magnetization configuration compatible to VTK software like ParaView

To save certain magnetization configurations like the result of the relaxation in the section above you can simply call the MicroMagnum storage methods with the field of interest.

.. code-block:: python

    # write the current magnetization to an OMF file
    writeOMF("filename.omf", solver.state.M)

    # write the current magnetization to a VTK file
    writeVTK("filename.vtk", solver.state.M)

If you want to save data along the simulation to get information of the time evolution of the system you can add step handlers to the solver instance.

.. code-block:: python

    # Save the field as ODT every 10th integration step
    solver.addStepHandler(
        DataTableLog("odtdata.odt"),
        condition.EveryNthStep(10))

    # Save the field as VTK every 10th integration step in a subfolder(here:"vtk") 
    solver.addStepHandler(
        VTKStorage("vtk", "M"),
        condition.EveryNthStep(10))

The second argument of the addStepHandler function takes a condition. The step handler is only executed if this condition is true.
