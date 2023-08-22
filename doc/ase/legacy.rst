.. _removed_features:

Legacy functionality
====================

Sometimes features are removed from ASE.  These features may still be
useful, and can be found in version history, only we can't support
them.

Common reasons for removing a feature are that the feature is
unused, buggy or broken, untested, undocumented, difficult to
maintain, or has possible security problems.

Below is a list of removed features.

========================================================== ========== =============================
Removed in 3.23.0                                          MR         Comment
========================================================== ========== =============================
``ase.calculators.vasp.vasp_auxiliary.xdat2traj``          :mr:`2948` Use ``ase.io``
``ase.io.gaussian_reader``                                 :mr:`2329` No tests or documentation
========================================================== ========== =============================

========================================================== ========== =============================
Removed in 3.21.0                                          MR         Comment
========================================================== ========== =============================
``ase.calculators.ase_qmmm_manyqm``                        :mr:`2092` Has docs but lacks real tests
``ase.build.voids``                                        :mr:`2078`
Unused code in ``ase.transport.tools``                     :mr:`2077`
``ase.io.iwm``                                             :mr:`2064`
``ase.visualize.primiplotter``                             :mr:`2060` Moved to asap3
``ase.visualize.fieldplotter``                             :mr:`2060` Moved to asap3
``ase.io.plt``                                             :mr:`2057`
========================================================== ========== =============================

========================================================== ========== =============================
Removed in 3.20.0                                          MR         Comment
========================================================== ========== =============================
dacapo-netcdf in ``ase.io.dacapo``                         :mr:`1892`
``ase.build.adsorb``                                       :mr:`1845`
Unused code in ``ase.utils.ff``                            :mr:`1844`
``ase.utils.extrapolate``                                  :mr:`1808` Moved to GPAW
``ase.calculators.dacapo``                                 :mr:`1721`
``ase/data/tmgmjbp04n.py``                                 :mr:`1720`
``ase/data/tmfp06d.py``                                    :mr:`1720`
``ase/data/gmtkn30.py``                                    :mr:`1720`
``ase/data/tmxr200x_tm3r2008.py``                          :mr:`1720`
``ase/data/tmxr200x_tm2r2007.py``                          :mr:`1720`
``ase/data/tmxr200x_tm1r2006.py``                          :mr:`1720`
``ase/data/tmxr200x.py``                                   :mr:`1720`
``ase.spacegroup.findsym``                                 :mr:`1692` Use spglib
``ase.calculators.jacapo``                                 :mr:`1604`
========================================================== ========== =============================
