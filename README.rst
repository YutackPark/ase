ASE - a Python library for working with atoms
=============================================

Webpage: http://wiki.fysik.dtu.dk/ase


Requirements
------------

* Python_ 2.6-3.5
* NumPy_ (base N-dimensional array package)

Optional:

* SciPy_ (library for scientific computing)
* PyGTK_ and Matplotlib_ (for ``ase-gui``)


Installation
------------

Add ``/path/to/ase`` to your $PYTHONPATH environment variable and add
``/path/to/ase/tools`` to $PATH.
    
    
Testing
-------

Please run the tests::
    
    $ cd /path/to/ase
    $ python setup.py test  # takes 1 min.

and send us the output if there are failing tests.


Contact
-------

* Mailing lists: ase-users_ and ase-developers_
* IRC_: #ase on freenode.net

Please send us bug-reports, patches, code, ideas and questions.


Example
-------

Geometry optimization of hydrogen molecule with NWChem:
    
>>> from ase import Atoms
>>> from ase.optimize import BFGS
>>> from ase.calculators.nwchem import NWChem
>>> from ase.io import write
>>> h2 = Atoms('H2',
               positions=[[0, 0, 0],
                          [0, 0, 0.7]])
>>> h2.calc = NWChem(xc='PBE')
>>> opt = BFGS(h2, trajectory='h2.traj')
>>> opt.run(fmax=0.02)
BFGS:   0  19:10:49    -31.435229     2.2691
BFGS:   1  19:10:50    -31.490773     0.3740
BFGS:   2  19:10:50    -31.492791     0.0630
BFGS:   3  19:10:51    -31.492848     0.0023
>>> write('H2.xyz', h2)
>>> h2.get_potential_energy()  # ASE's units are eV and Å
-31.492847800329216

::
    
    $ ase-gui h2.traj
    
    
.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _Matplotlib: http://matplotlib.org/
.. _pygtk: http://www.pygtk.org/
.. _ase-users: https://listserv.fysik.dtu.dk/mailman/listinfo/ase-users
.. _ase-developers: https://listserv.fysik.dtu.dk/mailman/listinfo/ase-developers
.. _IRC: http://webchat.freenode.net/?randomnick=0&channels=ase