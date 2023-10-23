.. module:: ase.calculators.espresso

========
Espresso
========

.. image:: ../../static/espresso.png

`Quantum ESPRESSO <http://www.quantum-espresso.org>`_ (QE) is an integrated
suite of Open-Source computer codes for electronic-structure calculations and
materials modeling at the nanoscale. It is based on density-functional
theory, plane waves, and pseudopotentials.

The ASE calculator is an interface to the ``pw.x`` executable.

Calculations using the Espresso calculator can be run by defining an 
``EspressoProfile`` and setting up an ``Espresso`` calculator with it::

  from ase.calculators.espresso import Espresso, EspressoProfile

  espresso_profile = EspressoProfile(["pw.x"])
  calc = Espresso(profile=espresso_profile)

The above ``calc`` object will run the ``pw.x`` executable on 1 core. 
The argument to the ``EspressoProfile`` object is a list of tokens that
can be executed in the shell to run QE as desired. An example of such a 
list would be::

  ["mpirun", "-np", "16", "pw.x"] # Run QE on 16 cores

Lists like the above can be obtained conveniently using string splitting.
Below we provide a few examples of QE run commands::

  "mpirun -np 16 pw.x" # Run QE on 16 cores, with default parallelism.
  "mpirun -np 64 pw.x -nk 2" # Run QE on 64 cores, parallel over 2 kpoints.

  # As above, diagonalize using Scalapack/Elpa 
  # (if your QE was installed with support) on an 8x8
  # grid of processors.
  "mpirun -np 64 pw.x -ndiag 64"

For calculations one would define the profile and the calculator as::
  
  from ase.calculators.espresso import Espresso, EspressoProfile

  NSLOTS = 64
  espresso_profile = EspressoProfile(f"mpirun -np {NSLOTS} pw.x".split())

  # This line is incomplete, since the user must also provide calculation
  # inputs before a successful calculation can be run.
  calc = Espresso(profile=espresso_profile)

Any calculation will need pseudopotentials for the elements involved. The
directory for the pseudopotential files can be set with the ``pseudo_dir``
parameter, otherwise QE will look in ``$ESPRESSO_PSEUDO`` if it is set
as an environment variable if set; otherwise ``$HOME/espresso/pseudo/`` is
used. The pseudopotentials are assigned for each element as a dictionary::

    pseudopotentials = {'Na': 'na_pbe_v1.5.uspp.F.UPF',
                        'Cl': 'cl_pbe_v1.4.uspp.F.UPF'}

A recommended list of pseudopotentials is the SSSP curated set, which can be
found on `Materials Cloud <https://www.materialscloud.org
/discover/sssp/table/efficiency>`_.

A simple calculation can be set up like so::

    from ase.build import bulk
    from ase.calculators.espresso import Espresso, EspressoProfile
    from ase.constraints import ExpCellFilter
    from ase.optimize import LBFGS

    rocksalt = bulk('NaCl', crystalstructure='rocksalt', a=6.0)

    # Pseudopotentials from SSSP Efficiency v1.3.0
    pseudopotentials = {'Na': 'na_pbe_v1.5.uspp.F.UPF',
                        'Cl': 'cl_pbe_v1.4.uspp.F.UPF'}
                        
    calc = Espresso(profile=EspressoProfile("mpirun -np 4 pw.x".split()),
                    ecutwfc=25, # This is unconverged and used for this example
                    pseudopotentials=pseudopotentials,
                    pseudo_dir=path_to_pseudopotentials,
                    tstress=True, tprnfor=True, kpts=(3, 3, 3))
    rocksalt.calc = calc

    filt = ExpCellFilter(rocksalt)
    opt = LBFGS(filt)
    opt.run(fmax=0.005)

    # cubic lattic constant
    print((8*rocksalt.get_volume()/len(rocksalt))**(1.0/3.0))

Parameters
==========

The calculator will interpret most documented options for ``pw.x``
found at the `PW input description <https://www.quantum-espresso.org/
Doc/INPUT_PW.html>`_, current exceptions being (at least) the ``HUBBARD``, 
``&FCP``, ``&RISM`` sections.

All parameters must be given in QE units, usually Ry or atomic units
in line with the documentation. ASE does not add any defaults over the
defaults of QE.

Parameters can be given as keywords and the calculator will put them into
the correct section of the input file. The calculator also accepts a keyword
argument ``input_data`` which is a dict, parameters may be put into sections
in ``input_data``, but it is not necessary::

    input_data = {
        'system': {
          'ecutwfc': 64,
          'ecutrho': 576}
        'disk_io': 'low'}  # automatically put into 'control'

    calc = Espresso(profile=my_predefined_profile,
                    pseudopotentials=pseudopotentials,
                    # Optionally, if the ESPRESSO_PSEUDO env var isn't set
                    pseudo_dir=path_to_pseudopotentials,
                    tstress=True, tprnfor=True,  # kwargs added to parameters
                    input_data=input_data)

Some parameters are used by ASE, or have additional meaning:

 * ``kpts``, is used to specify the k-point sampling
   * If ``kpts`` is a tuple (or list) of 3 integers ``kpts=(int, int, int)``, it is interpreted  as the dimensions of a Monkhorst-Pack grid.
   * If ``kpts`` is set to ``None``, only the Γ-point will be included and QE will use routines optimized for Γ-point-only calculations. Compared to Γ-point-only calculations without this optimization (i.e. with ``kpts=(1, 1, 1)``), the memory and CPU requirements are typically reduced by half.
   * If ``kpts`` is a dict, it will either be interpreted as a path in the Brillouin zone (see ase.dft.kpoints.bandpath) if it contains the 'path' keyword, otherwise it is converted to a Monkhorst-Pack grid (see ase.calculators.calculator.kpts2sizeandoffsets)
 * ``koffset=(0, 0, 0)`` set to 0 or 1 to displace the kpoint grid by a half
   cell in that direction. Also accepts ``True`` and ``False``.
 * ``kspacing=0.1`` sets the minimum distance between kpoints in reciprocal
   space.
 * ``nspin=2`` if any atom has a magnetic moment spin is turned on
   automatically.

Any ``FixAtoms`` or ``FixCartesian`` constraints are converted to Espresso
constraints (for dynamic calculations).


Alternative Calculators
=======================

There are several other QE ``Calculator`` implementations based on ``ase``
that provide a number of extra features:

 - http://jochym.github.io/qe-util/
 - https://github.com/vossjo/ase-espresso

Espresso Calculator Class
=========================

.. autoclass:: ase.calculators.espresso.Espresso

