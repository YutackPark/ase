.. module:: ase.calculators
   :synopsis: Energy, force and stress calculators.

.. _calculators:

===========
Calculators
===========

For ASE, a calculator is a black box that can take atomic numbers and
atomic positions from an :class:`~ase.Atoms` object and calculate the
energy and forces and sometimes also stresses.

In order to calculate forces and energies, you need to attach a
calculator object to your atoms object:

>>> atoms = read('molecule.xyz')
>>> e = atoms.get_potential_energy()  # doctest: IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/jjmo/ase/atoms/ase.py", line 399, in get_potential_energy
    raise RuntimeError('Atoms object has no calculator.')
RuntimeError: Atoms object has no calculator.
>>> from ase.calculators.abinit import Abinit
>>> calc = Abinit(...)
>>> atoms.calc = calc
>>> e = atoms.get_potential_energy()
>>> print(e)
-42.0

Here we attached
an instance of the :mod:`ase.calculators.abinit` class and then
we asked for the energy.


.. _supported calculators:

Supported calculators
=====================

The calculators can be divided in four groups:

1) Abacus_, ALIGNN_, AMS_, Asap_, BigDFT_, CHGNet_, DeePMD-kit_, DFTD3_, DFTD4_, DFTK_, FLEUR_, GPAW_, Hotbit_, M3GNet_, MACE_, TBLite_, and XTB_
   have their own native or external ASE interfaces.

2) ABINIT, AMBER, CP2K, CASTEP, deMon2k, DFTB+, ELK, EXCITING, FHI-aims, GAUSSIAN,
   Gromacs, LAMMPS, MOPAC, NWChem, Octopus, ONETEP, PLUMED, psi4, Q-Chem, Quantum ESPRESSO, SIESTA,
   TURBOMOLE and VASP, have Python wrappers in the ASE package, but the actual
   FORTRAN/C/C++ codes are not part of ASE.

3) Pure python implementations included in the ASE package: EMT, EAM,
   Lennard-Jones, Morse and HarmonicCalculator.

4) Calculators that wrap others, included in the ASE package:
   :class:`ase.calculators.checkpoint.CheckpointCalculator`,
   the :class:`ase.calculators.loggingcalc.LoggingCalculator`,
   the :class:`ase.calculators.mixing.LinearCombinationCalculator`,
   the :class:`ase.calculators.mixing.MixedCalculator`,
   the :class:`ase.calculators.mixing.SumCalculator`,
   the :class:`ase.calculators.mixing.AverageCalculator`,
   the :class:`ase.calculators.socketio.SocketIOCalculator`,
   the :ref:`Grimme-D3 <grimme>` potential, and the qmmm calculators
   :class:`~ase.calculators.qmmm.EIQMMM`,  and :class:`~ase.calculators.qmmm.SimpleQMMM`.

========================================= ===========================================
name                                      description
========================================= ===========================================
Abacus_                                   DFT supporting both pw and lcao basis
ALIGNN_                                   Atomistic Line Graph Neural Network force field
AMS_                                      Amsterdam Modeling Suite
Asap_                                     Highly efficient EMT code
BigDFT_                                   Wavelet based code for DFT
CHGNet_                                   Universal neural network  potential for charge-informed atomistics
DeePMD-kit_                               A deep learning package for many-body potential energy representation
DFTD3_                                    London-dispersion correction
DFTD4_                                    Charge-dependent London-dispersion correction
DFTK_                                     Plane-wave code for DFT and related models
FLEUR_                                    Full Potential LAPW code
GPAW_                                     Real-space/plane-wave/LCAO PAW code
Hotbit_                                   DFT based tight binding
M3GNet_                                   Materials 3-body Graph Network universal potential
MACE_                                     Many-body potential using higher-order equivariant message passing
TBLite_                                   Light-weight tight-binding framework
XTB_                                      Semiemprical extended tight-binding program package
:mod:`~ase.calculators.abinit`            Plane-wave pseudopotential code
:mod:`~ase.calculators.amber`             Classical molecular dynamics code
:mod:`~ase.calculators.castep`            Plane-wave pseudopotential code
:mod:`~ase.calculators.cp2k`              DFT and classical potentials
:mod:`~ase.calculators.demon`             Gaussian based DFT code
:mod:`~ase.calculators.demonnano`         DFT based tight binding code
:mod:`~ase.calculators.dftb`              DFT based tight binding
:mod:`~ase.calculators.dmol`              Atomic orbital DFT code
:mod:`~ase.calculators.eam`               Embedded Atom Method
elk                                       Full Potential LAPW code
:mod:`~ase.calculators.espresso`          Plane-wave pseudopotential code
:mod:`~ase.calculators.exciting`          Full Potential LAPW code
:mod:`~ase.calculators.aims`              Numeric atomic orbital, full potential code
:mod:`~ase.calculators.gamess_us`         Gaussian based electronic structure code
:mod:`~ase.calculators.gaussian`          Gaussian based electronic structure code
:mod:`~ase.calculators.gromacs`           Classical molecular dynamics code
:mod:`~ase.calculators.gulp`              Interatomic potential code
:mod:`~ase.calculators.harmonic`          Hessian based harmonic force-field code
:mod:`~ase.calculators.kim`               Classical MD with standardized models
:mod:`~ase.calculators.lammps`            Classical molecular dynamics code
:mod:`~ase.calculators.mixing`            Combination of multiple calculators
:mod:`~ase.calculators.mopac`             Semiempirical molecular orbital code
:mod:`~ase.calculators.nwchem`            Gaussian based electronic structure code
:mod:`~ase.calculators.octopus`           Real-space pseudopotential code
:mod:`~ase.calculators.onetep`            Linear-scaling pseudopotential code
:mod:`~ase.calculators.openmx`            LCAO pseudopotential code
:mod:`~ase.calculators.orca`              Gaussian based electronic structure code
:mod:`~ase.calculators.plumed`            Enhanced sampling method library
:mod:`~ase.calculators.psi4`              Gaussian based electronic structure code
:mod:`~ase.calculators.qchem`             Gaussian based electronic structure code
:mod:`~ase.calculators.siesta`            LCAO pseudopotential code
:mod:`~ase.calculators.turbomole`         Fast atom orbital code
:mod:`~ase.calculators.vasp`              Plane-wave PAW code
:mod:`~ase.calculators.emt`               Effective Medium Theory calculator
lj                                        Lennard-Jones potential
morse                                     Morse potential
:mod:`~ase.calculators.checkpoint`        Checkpoint calculator
:mod:`~ase.calculators.socketio`          Socket-based interface to calculators
:mod:`~ase.calculators.loggingcalc`       Logging calculator
:mod:`~ase.calculators.dftd3`             DFT-D3 dispersion correction calculator
:class:`~ase.calculators.qmmm.EIQMMM`     Explicit Interaction QM/MM
:class:`~ase.calculators.qmmm.SimpleQMMM` Subtractive (ONIOM style) QM/MM
========================================= ===========================================

.. index:: D3, Grimme
.. _grimme:

.. note::

    A Fortran implemetation of the Grimme-D3 potential, that can be used as
    an add-on to any ASE calculator, can be found here:
    https://gitlab.com/ehermes/ased3/tree/master.

The calculators included in ASE are used like this:

>>> from ase.calculators.abc import ABC
>>> calc = ABC(...)

where ``abc`` is the module name and ``ABC`` is the class name.

.. _Abacus: https://gitlab.com/1041176461/ase-abacus
.. _ALIGNN: https://github.com/usnistgov/alignn?tab=readme-ov-file#alignnff
.. _AMS: https://www.scm.com/doc/plams/examples/AMSCalculator/ASECalculator.html#asecalculatorexample
.. _Asap: https://wiki.fysik.dtu.dk/asap
.. _BigDFT: https://l_sim.gitlab.io/bigdft-suite/tutorials/Interoperability-Simulation.html#ASE-Interoperability
.. _CHGNet: https://github.com/CederGroupHub/chgnet/blob/e2a2b82bf2c64e5a3d39cd75d0addfa864a2771a/chgnet/model/dynamics.py#L63
.. _GPAW: https://wiki.fysik.dtu.dk/gpaw
.. _Hotbit: https://github.com/pekkosk/hotbit
.. _DFTK: https://dftk.org
.. _DeePMD-kit: https://github.com/deepmodeling/deepmd-kit
.. _DFTD4: https://github.com/dftd4/dftd4/tree/main/python
.. _DFTD3: https://dftd3.readthedocs.io/en/latest/api/python.html#module-dftd3.ase
.. _FLEUR: https://github.com/JuDFTteam/ase-fleur
.. _M3GNet: https://matgl.ai/matgl.ext.html#class-matglextasem3gnetcalculatorpotential-potential-state_attr-torchtensor--none--none-stress_weight-float--10-kwargs
.. _MACE: https://mace-docs.readthedocs.io/en/latest/guide/ase.html
.. _TBLite: https://tblite.readthedocs.io/en/latest/users/ase.html
.. _XTB: https://xtb-python.readthedocs.io/en/latest/ase-calculator.html

Calculator configuration
========================

As of November 2023, there are two ways in which a calculator can be implemented:

  * a modern way -- subclassing a Calculator class from :class:`ase.calculators.genericfileio.GenericFileIOCalculator`
    (calculators implemented in such a way are ABINIT, FHI-Aims, Quantum ESPESSO, EXCITING, Octopus and Orca; there are
    plans to gradually rewrite the remaining calculators as well);
  * a somewhat conservative way, subclassing it from :class:`ase.calculators.calculator.FileIOCalculator`.

The calculators that are implemented in the modern way can be configured using the config file. It should have a `.ini`
format and reside in a place specified by ``ASE_CONFIG_PATH`` environmental variable. If the variable is not set, then the
default path is used, which is ``~/.config/ase/config.ini``.

The config file should have a ``[parallel]`` section, which defines the machine-specific parallel environment, and the
calculator sections, that define the machine-specific calculator parameters, like binary and pseudopotential locations.
The parallel section should have a ``binary`` option, which should point to the name of the parallel runner binary file,
like `mpirun` or `mpiexec`. Then the Calculator class instance can be initialized with ``parallel=True`` keyword. This
allows running the calculator code in parallel. The additional keywords to the parallel runner can be specified with
``parallel_info=<dict>`` keyword, which gets translated to the list of flags and their values passed to the parallel
runner. Translation keys can be specified in the ``[parallel]`` section with the syntax ``key_kwarg_trans = command``
e.g if ``nprocs_kwarg_trans = -np`` is specified in the config file, then the key ``nprocs`` will be
translated to ``-np``. Then `nprocs` can be specified in ``parallel_info`` and will be translated to `-np` when the command is build.

The example of a config file is as follows::

    [parallel]
    binary = mpirun
    nprocs_kwarg_trans = -np

    [espresso]
    binary = pw.x
    pseudo_path = /home/ase/upf_pseudos

Then the `espresso` calculator can be invoked in the following way::

    >>> from ase.build import bulk
    >>> from ase.calculators.espresso import Espresso
    >>> espresso = Espresso(
                       input_data = {
                            'system': {
                               'ecutwfc': 60,
                            }},
                       pseudopotentials = {'Si': 'si_lda_v1.uspp.F.UPF'},
                       parallel=True,
                       parallel_info={'nprocs': 4}
                       )
    >>> si = bulk('Si')
    >>> si.calc = espresso
    >>> si.get_potential_energy()
    -244.76638508140397

Here espresso ran in parallel with 4 processes and produced a correct result.

Calculator keywords
===================

Example for a hypothetical ABC calculator:

.. class:: ABC(restart=None, ignore_bad_restart_file=False, label=None,
               atoms=None, parameters=None, command='abc > PREFIX.abc',
               xc=None, kpts=[1, 1, 1], smearing=None,
               charge=0.0, nbands=None, **kwargs)

   Create ABC calculator

   restart: str
       Prefix for restart file.  May contain a directory.  Default
       is None: don't restart.
   ignore_bad_restart_file: bool
       Ignore broken or missing restart file.  By default, it is an
       error if the restart file is missing or broken.
   label: str
       Name used for all files.  May contain a directory.
   atoms: Atoms object
       Optional Atoms object to which the calculator will be
       attached.  When restarting, atoms will get its positions and
       unit-cell updated from file.
   command: str
       Command used to start calculation.  This will override any value
       in an :envvar:`ASE_ABC_COMMAND` environment variable.
   parameters: str
       Read parameters from file.
   xc: str
       XC-functional (``'LDA'``, ``'PBE'``, ...).
   kpts:
       Brillouin zone sampling:

       * ``(1,1,1)``: Gamma-point
       * ``(n1,n2,n3)``: Monkhorst-Pack grid
       * ``(n1,n2,n3,'gamma')``: Shifted Monkhorst-Pack grid that includes
         `\Gamma`
       * ``[(k11,k12,k13),(k21,k22,k23),...]``: Explicit list in units of the
         reciprocal lattice vectors
       * ``kpts=3.5``: `\vec k`-point density as in 3.5 `\vec k`-points per
         Å\ `^{-1}`.
   smearing: tuple
       The smearing of occupation numbers.  Must be a tuple:

       * ``('Fermi-Dirac', width)``
       * ``('Gaussian', width)``
       * ``('Methfessel-Paxton', width, n)``, where `n` is the order
         (`n=0` is the same as ``'Gaussian'``)

       Lower-case names are also allowed.  The ``width`` parameter is
       given in eV units.
   charge: float
      Charge of the system in units of `|e|` (``charge=1`` means one
      electron has been removed).  Default is ``charge=0``.
   nbands: int
      Number of bands.  Each band can be occupied by two electrons.

Not all of the above arguments make sense for all of ASE's
calculators.  As an example, Gromacs will not accept DFT related
keywords such as ``xc`` and ``smearing``.  In addition to the keywords
mentioned above, each calculator may have native keywords that are
specific to only that calculator.

Keyword arguments can also be set or changed at a later stage using
the :meth:`set` method:

.. method:: set(key1=value1, key2=value2, ...)


.. toctree::

   eam
   emt
   abinit
   amber
   castep
   cp2k
   crystal
   demon
   demonnano
   dftb
   dmol
   espresso
   exciting
   FHI-aims
   fleur
   gamess_us
   gaussian
   gromacs
   gulp
   harmonic
   socketio/socketio
   jacapo
   kim
   lammps
   mopac
   nwchem
   octopus
   onetep
   openmx
   orca
   plumed
   psi4
   qchem
   siesta
   turbomole
   vasp
   qmmm
   checkpointing
   mixing
   loggingcalc
   dftd3
   others
   test
   ace
