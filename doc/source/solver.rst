========
 Solver 
========

.. currentmodule:: magnum

Solver
------

.. autoclass:: Solver

   .. automethod:: solve

TODO methods

Evolvers
^^^^^^^^

Currenty, three types of evolvers for the time integration of the LLG
equation are available.  They are selected using the 'evolver' argument
at the time of creation of the Solver object:

- Euler evolver with constant time steps ("``euler``")

  .. code-block:: python
  
    # Create solver with Euler evolver
    s = create_solver(evolver="euler", time_step=1e-14)

  The ``time_step`` parameter is optional, default is 1e-14.

- Runge-Kutta-Fehlberg evolver (RKF45) with variable time steps ("``rkf45``")

  .. code-block:: python
  
    # Create solver with RKF45 evolver
    s = create_solver(evolver="rkf45", eps_abs=1e-3, eps_rel=1e-4)

  The ``eps_abs`` and ``eps_rel`` parameters are optional with the defaults
  being as shown in the example.

- CVode implicit evolver with variable timesteps ("``cvode``")

  .. code-block:: python
  
    # Create solver with cvode evolver
    s = create_solver(evolver="cvode", eps_abs=1e-3, eps_rel=1e-4, 
                      step_size=1e-12, newton_method=False)

  The ``eps_abs``, ``eps_rel``, ``step_size`` and ``newton_method`` parameters are optional with the defaults
  being as shown in the example.

The RK45 evolver is the default evolver when the 'evolver' option is ommited.

+------------------------------+----------------+------------------------------------------+
| Name                         | Evolver id     | Options (defaults)                       |
+==============================+================+==========================================+
| Euler                        | ``euler``      | ``step_size`` (:math:`1\cdot 10^{-14}`)  |
+------------------------------+----------------+------------------------------------------+
| Runge-Kutta-Fehlberg 45      | ``rkf45``      | ``eps_abs`` (:math:`1\cdot 10^{-3}`),    |
|                              |                | ``eps_rel`` (:math:`1\cdot 10^{-4}`)     |
+------------------------------+----------------+------------------------------------------+
| CVode implicit evolver       | ``cvode``      | ``eps_abs`` (:math:`1\cdot 10^{-3}`),    |
|                              |                | ``eps_rel`` (:math:`1\cdot 10^{-4}`),    |
|                              |                | ``step_size`` (:math:`1\cdot 10^{-12}`), |
|                              |                | ``newton_method`` (``False``)            |
+------------------------------+----------------+------------------------------------------+


CVode evolver
"""""""""""""

CVode uses two iteration methods, fuctional and Newton.
The functional method is very fast and more stable than Runge-Kutta.
The Newton method is very slow and very stable.
Information about the evolver are available on:
http://computation.llnl.gov/casc/sundials/documentation/documentation.html
The relax condition does not work with CVode, Runge-Kutta should be used.

Conditions
----------

Objects of the Condition class check whether some condition for a magnetic
state is true.  For example, the Condition.relaxed() condition is true
when the time deriviative of the magnetization :math:`d\mathbf{M}/dt`
of a given simulation state is small enough. This condition are used as
abort criterions for simulations. Another application is to specify when
step handlers (see :class:`StepHandler`) are activated during
a simulations.

Conditions can be negated, or-combined and and-combined using operators.

.. autoclass:: magnum.solver.condition

   .. automethod:: Condition.__init__
   .. automethod:: Condition.check

   The following factory methods provide some useful predefined conditions.

     *Examples*:
   
       .. code-block:: python

          # Condition that checks if magnetization is relaxed
          c1 = Condition.relaxed()
          # Condition that checks if simulation time is greater than 2ns
          c2 = Condition.timeGreaterEq(2e-9)
          # Condition that checks if simulation time falls between time interval [2ns, 6ns]
          c3 = Condition.timeBetween(2e-9, 6e-9)

     .. automethod:: Condition.everyNthStep
     .. automethod:: Condition.afterNthStep
     .. automethod:: Condition.timeGreaterEq
     .. automethod:: Condition.timeBetween
     .. automethod:: Condition.relaxed
     .. automethod:: Condition.always
     .. automethod:: Condition.never

   Conditions can be combined using the ~ (__invert__), | (__or__) and &
   (__and__) operators that are part of the Python language.
   
   *Examples*:
      
      .. code-block:: python
   
         # Condition that checks if magnetization is either relaxed or simulation time is greater than 5ns.
         c1 = Condition.relaxed() or Condition.timeGreaterEq(5e-9)
         # Condition that checks if simulation time is smaller than 4ns
         c2 = ~Condition.timeGreaterEq(5e-9)
   
   .. automethod:: Condition.__and__
   .. automethod:: Condition.__or__
   .. automethod:: Condition.__invert__

Step handlers
-------------

Step handlers are added to the solver object by calling :func:`Solver.addStepHandler`.

The base class of all step handlers is the abstract class StepHandler:

.. autoclass:: StepHandler

   .. automethod:: StepHandler.handle
   .. automethod:: StepHandler.done

Logging step handlers
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: DataTableLog

   Example OOMMF data table (.odt) log file that is generated by :class:`DataTableLog`:

   .. code-block:: bash

      # ODT 1.0
      # Table Start
      # Title: (no title)
      #
      ## Generated by magnum.DataTableStepHandler.
      ## Number of columns: 7
      #
      # Columns:   {time}                    {step size}               {average magn. x}         {average magn. y}         {average magn. z}         {stray-field energy}      {exchange-field energy}    
      # Units:     {s}                       {s}                       {A/m}                     {A/m}                     {A/m}                     {J}                       {J}                       
      0.0                       1.0000000000000001e-15    773763.05520921084    99865.913350068877    0.51014508223702026    5.425877056278558e-19    8.8083062896386378e-20    
      7.190533493683745e-12     1.0932486392938052e-12    772611.96576462197    103833.22777842356    -7219.4953021688525   5.4851562239571601e-19    9.0258813551580614e-20    
      2.3403744897262799e-11    1.5074283534957295e-12    760973.57366955292    139366.78750870723    -22539.08658192489    6.0677999378414166e-19    1.0995487038693322e-19    
      3.9428649475373533e-11    1.85956428734661e-12      733162.57517834462    204774.35569038574    -35631.238305689978    7.397106785586106e-19    1.4440096479070547e-19    
      ...
      ...
      ...
      9.6434123993105961e-10    8.9861003891477315e-13    -777442.58149938367    -58344.913010609191   6238.8726882989358    7.5244095099684009e-19    1.2958402944299087e-19    
      9.7402535807940305e-10    1.2062784047071156e-12    -783542.17931054463    -34240.439240366963   20335.17015782345     8.0952409880500454e-19    1.0968481551637731e-19    
      9.8338862949946303e-10    9.6173440992413807e-13    -789079.08096694865    8492.6521512550589    30227.519883341236    8.7514283220039697e-19    8.8932130375339342e-20    
      9.9374164359195942e-10    9.0429246884428303e-13    -789960.99319875205    68987.397611624358    34958.637897265799    9.1911429456896382e-19    7.6963987580309788e-20    
      # Table End

.. autoclass:: ScreenLog

   Example output on the console:

   .. code-block:: bash  
      
      # time: t (s)   step size: h (s)   average magn. x: Mx (A/m)   average magn. y: My (A/m)   average magn. z: Mz (A/m)   
      t=0.0, h=1.0000000000000001e-15, Mx=720260.46511223307, My=298341.65158440446, Mz=0.0
      t=1.3374476956998866e-10, h=1.8554788708302094e-12, Mx=759676.44090676622, My=157975.1496951601, Mz=3802.0657114495889
      t=2.9960043454703471e-10, h=1.9145468033064472e-12, Mx=770546.01143940352, My=111393.9534971742, Mz=717.70498717729379
      ...
      # done

Both the classes :class:`DataTableLog` and :class:`ScreenLog` have the following member functions:

.. method:: addColumn([(id, desc="(no descr.)", unit="unknown", fmt="%r")+], func)

   Adds one or more columns to the log. Each tuple represents one column. The last argument specifies a function that returns
   for a state object a tuple of <number of columns> scalar entries. Example:

   .. code-block:: python

      log = ScreenLog()
      log.addColumn(
          # description of one table column
          ("t", "simulation time", "s", "%r"),    
          # function that maps a state to its current simulation time
          lambda state: state.t

      log.addColumn(
          # description of first table columns
          ("t", "simulation time", "s", "%r"),    
          # description of another table column
          ("h", "step size", "s", "%r"),          
          # function that maps a state to its current simulation time and step size
          lambda state: (state.t, state.h)        
      )

   The first tuple entry specifies the variable name associated with the
   column. The second entry is a description of the variable, the third
   names the unit of the variable. The last entry specifies the format
   which is used to convert the scalar that is returned by the mapping
   function to a string. Only the first entry is mandatory. If only the first entry is specified,
   the tuple may be ommited:

   .. code-block:: python

      log.addColumn(("t",), lambda state: state.t)   # this line...
      log.addColumn("t", lambda state: state.t)      # ...is equal to this line.

      # another example:
      log.addColumn("t", "h", lambda state: state.t, state.h)

.. method:: addTimeColumn()

   Adds a simulation time column. As this column is added by default, this function is usually not called by the user.

.. method:: addStepSizeColumn()

   Adds a step size column. This column is added by default.

.. method:: addAverageMagColumn()

   Adds three columns to the log with the average magnetization in x, y and z-direction. Added by default.

.. method:: addEnergyColumn()

   TODO addEnergyColumn

Examples: Log simulation in a text file 'test.odt'. By default, the
simulation time, the time step and <Mx>, <My>, <Mz> is included resulting
in five table columns.

.. code-block:: python

   solver = create_solver(...)
   solver.addStepHandler(DataTableLog("test.odt")
   solver.solve(...)

Using the 'addColumn' and related functions one can add additional
columns to the log file:

.. code-block:: python
   
   log = DataTableLog("test.odt")    # (or: log = ScreenLog() for screen output)
   log.addEnergyColumn("exchange")   # adds column with E_exchange scalars
   log.addEnergyColumn("stray")      # adds column with E_stray scalars
   log.addTimeColumn()               # add (another) column with simulation time

   # another way to add a time column
   log.addColumn(("t", "time", "%r", "s"), lambda state: state.t)

   # Add two columns with vortex core x/y position using the vortex tool box.
   # (with (50nm, 50nm) as the origin).
   log.addColumn(("Vx", "vortex-core-x", "%r", "m"),
                 ("Vy", "vortex-core-y", "%r", "m"),
                 lambda state: vortex.findCore(solver, 50e-9, 50e-9))

   solver = create_solver(...)
   solver.addStepHandler(log)
   solver.solve(...)

File storage step handlers
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: VTKStorage
.. autoclass:: OOMMFStorage

.. method:: addComment(self, name, fn)

