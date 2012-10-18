=======
 World
=======

.. automodule:: magnum

In this section the world representation for simulation setups is described.

Simple world creation that is filled entirely with one material example:
  
.. code-block:: python
  
  # Create a world that is entirely filled with Permalloy.

  # First, create a mesh to spatially discretize the world
  mesh = RectangularMesh((100,100,1), (4e-9, 4e-9, 10e-9))

  # Second, create the world from the mesh and the material 
    (in this case permalloy)
  world = World(mesh, Material.Py())

More complex worlds can contain different sub-regions ("bodies") that
are each associated with an ID, a geometry (a Shape) and a material:

.. code-block:: python

  # Create a world that is composed of "Bodies"
  mesh = RectangularMesh((100,100,1), (4e-9, 4e-9, 10e-9))
  world = World(
    mesh,
    Body("the_body_id", Material.Py(), Cuboid((25e-9, 25e-9, 0e-9), 
        (75e-9, 75e-9, 10e-9)))
    # (optionally more bodies, separated by commas)
  )

World
-----

The world class.

.. autoclass:: World

   .. autoattribute:: World.mesh
   .. autoattribute:: World.bodies
   .. automethod:: World.findBody

Bodies
------

The Body class.

.. autoclass:: Body

   .. autoattribute:: Body.material
   .. autoattribute:: Body.shape
   .. autoattribute:: Body.id

Shapes
------

TODO The Shape class

.. autoclass:: Shape
.. autoclass:: ImageShape
.. autoclass:: Cuboid
.. autoclass:: Cylinder
.. autoclass:: Everywhere

Materials
---------

Materials are used in connection with the Body class to initialize material parameters.

+-------------+---------+----------+--------+-----------------+
| Material    |    Ms   |    A     |    K   | anisotropy type |
+=============+=========+==========+========+=================+
| Permalloy   |   800e3 |  13e-12  |    0   |  uniaxial       |
+-------------+---------+----------+--------+-----------------+
| Iron        |  1700e3 |  21e-12  |   48e3 |  cubic          |
+-------------+---------+----------+--------+-----------------+
| Nickel      |   490e3 |   9e-12  | -5.7e3 |  cubic          |
+-------------+---------+----------+--------+-----------------+
| Cobalt(hcp) |  1400e3 |  30e-12  |  520e3 |  uniaxial       |
+-------------+---------+----------+--------+-----------------+

In the following the Material class which holds the material parameters for a material type is described.

.. autoclass:: Material

   .. automethod:: Material.get
   .. automethod:: Material.Py
   .. automethod:: Material.Fe
   .. automethod:: Material.Co
   .. automethod:: Material.Ni

Examples:

.. code-block:: python

    # Create permalloy with default parameters
    py  = Material.Py()   

    # create own material
    mat = Material({'id':'MyMaterial', 'Ms': 8e5, 'A':13e-12, 'alpha':0.01}) 

