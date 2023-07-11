.. module:: ase.calculators.orca

======
ORCA
======

`ORCA <https://orcaforum.kofo.mpg.de/app.php/portal>`_ is a computational chemistry code 
that can do SCF, (TD)DFT, semi-empirical potentials, MP2, CASSCF, Coupled Cluster
calculations, and more. 


It is closed source, but free for academic users. Register on the forum to receive 
a download link for the binaries, as well as access to the latest manual.


Many input examples are available at the 
`ORCA Input Library <https://sites.google.com/site/orcainputlibrary>`_.


.. highlight:: none

The :class:`ORCA` ASE-interface is very simple. Two keywords are defined::

  orcasimpleinput: str
      What you'd put after the "!" in an orca input file.

  orcablocks: str
      What you'd put in the "% ... end"-blocks.


The ASE-calculator also works with the 
:mod:`~ase.calculators.qmmm.EIQMMM`-calculator 
for QM/MM simulations (see :mod:`~ase.calculators.qmmm` for 
more info). 

Setup and usage
===============

.. highlight:: bash

The default command that ASE will use to start ORCA is
``orca PREFIX.inp > PREFIX.out``. 

For parallel calculations, you will need to specify the full path to the ORCA
executable. This can be done by defining an `OrcaProfile` like the example below:

  from ase.calculators.orca import OrcaProfile

  MyOrcaProfile = OrcaProfile(["/full/path/to/my/orca"])
  calc = ORCA(profile=MyOrcaProfile)

.. highlight:: python

ORCA decides which sub-processes to parallelize via MPI by itself, so you'll
almost always want a string in your ``orcablocks`` specifying the number of 
cores for the simulation, e.g.::

  from ase.calculators.orca import ORCA

  calc = ORCA(profile=MyOrcaProfile, 
              orcasimpleinput='B3LYP def2-TZVP'
              orcablocks='%pal nprocs 16 end'

for a B3LYP/def2-TZVP calculation on 16 cores. 

Class Definition
================

.. autoclass:: ORCA
