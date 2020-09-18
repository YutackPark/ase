import sys
import numpy as np

import ase.units as u
from ase.parallel import world, parprint, paropen
from ase.phonons import Phonons
from ase.vibrations import Vibrations
from ase.utils.timing import Timer
from ase.utils import convert_string_to_fd
from ase.dft import monkhorst_pack


class RamanBase:
    def __init__(self, atoms,  # XXX do we need atoms at this stage ?
                 *args,
                 name='raman',
                 exext='.alpha',
                 txt='-',
                 verbose=False,
                 comm=world,
                 **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        exext: string
          Extension for excitation filenames
        txt:
          Output stream
        verbose:
          Verbosity level of output
        comm:
          Communicator, default world
        """
        kwargs['name'] = name
        self.exname = kwargs.pop('exname', name)

        super().__init__(atoms, *args, **kwargs)

        self.exext = exext

        self.timer = Timer()
        self.txt = convert_string_to_fd(txt)
        self.verbose = verbose

        self.comm = comm

    def log(self, message, pre='# ', end='\n'):
        if self.verbose:
            self.txt.write(pre + message + end)
            self.txt.flush()


class StaticRamanCalculatorBase(RamanBase):
    """Base class for Raman intensities derived from
    static polarizabilities"""
    def __init__(self, atoms, exobj, *args, **kwargs):
        self.exobj = exobj
        super().__init__(atoms, *args, **kwargs)
        
    def calculate(self, atoms, filename, fd):
        # write forces
        super().calculate(atoms, filename, fd)
        # write static polarizability
        fname = filename.replace('.pckl', self.exext)
        np.savetxt(fname, self.exobj().calculate(atoms))
      

class StaticRamanCalculator(StaticRamanCalculatorBase, Vibrations):
    pass


class StaticRamanPhononCalculator(StaticRamanCalculatorBase, Phonons):
    pass


class RamanEvaluate(RamanBase):
    """Base class to evaluate Raman spectra from pre-computed data"""
    def __init__(self, atoms,  # XXX do we need atoms at this stage ?
                 *args,
                 exname=None,      # name for excited state calculations
                 **kwargs):
        """
        Parameters
        ----------
        atoms: ase Atoms object
        exname: string
            name for excited state calculations (defaults to name),
            used for reading excitations
        """
        super().__init__(atoms, *args, **kwargs)

        if exname is None:
            exname = kwargs.get('name', self.name)
        self.exname = exname
        self._read = False

        # xxx
        kwargs.pop('txt', None)
        kwargs.pop('exext', None)
        kwargs['name'] = kwargs.get('name', self.name)
        self.vibrations = Vibrations(atoms, *args, **kwargs)

    def get_energies(self):
        if not self._read:
            self.read()
        return self.om_Q

    def init_parallel_read(self):
        """Initialize variables for parallel read"""
        rank = self.comm.rank
        self.ndof = 3 * len(self.indices)
        myn = -(-self.ndof // self.comm.size)  # ceil divide
        self.slize = s = slice(myn * rank, myn * (rank + 1))
        self.myindices = np.repeat(self.indices, 3)[s]
        self.myxyz = ('xyz' * len(self.indices))[s]
        self.myr = range(self.ndof)[s]
        self.mynd = len(self.myr)

    def read(self, *args, **kwargs):
        """Read data from a pre-performed calculation."""
        self.timer.start('read')

        self.timer.start('vibrations')
        self.read_energies_and_modes(*args, **kwargs)
        self.timer.stop('vibrations')
        
        self.timer.start('excitations')
        self.init_parallel_read()
        self.read_excitations()
        self.timer.stop('excitations')

        self._read = True
        self.timer.stop('read')

    @staticmethod
    def m2(z):
        return (z * z.conj()).real

    def me_Qcc(self, *args, **kwargs):
        """Full matrix element

        Returns
        -------
        Matrix element in e^2 Angstrom^2 / eV
        """
        # Angstrom^2 / sqrt(amu)
        elme_Qcc = self.electronic_me_Qcc(*args, **kwargs)
        # Angstrom^3 -> e^2 Angstrom^2 / eV
        elme_Qcc /= u.Hartree * u.Bohr  # e^2 Angstrom / eV / sqrt(amu)
        return elme_Qcc * self.vib01_Q[:, None, None]

    def get_absolute_intensities(self, delta=0, **kwargs):
        """Absolute Raman intensity or Raman scattering factor

        Parameter
        ---------
        delta: float
           pre-factor for asymmetric anisotropy, default 0

        References
        ----------
        Porezag and Pederson, PRB 54 (1996) 7830-7836 (delta=0)
        Baiardi and Barone, JCTC 11 (2015) 3267-3280 (delta=5)

        Returns
        -------
        raman intensity, unit Ang**4/amu
        """
        alpha2_r, gamma2_r, delta2_r = self._invariants(
            self.electronic_me_Qcc(**kwargs))
        return 45 * alpha2_r + delta * delta2_r + 7 * gamma2_r

    def intensity(self, *args, **kwargs):
        """Raman intensity

        Returns
        -------
        unit e^4 Angstrom^4 / eV^2
        """
        m2 = Raman.m2
        alpha_Qcc = self.me_Qcc(*args, **kwargs)
        if not self.observation:  # XXXX remove
            """Simple sum, maybe too simple"""
            return m2(alpha_Qcc).sum(axis=1).sum(axis=1)
        # XXX enable when appropriate
        #        if self.observation['orientation'].lower() != 'random':
        #            raise NotImplementedError('not yet')

        # random orientation of the molecular frame
        # Woodward & Long,
        # Guthmuller, J. J. Chem. Phys. 2016, 144 (6), 64106
        alpha2_r, gamma2_r, delta2_r = self._invariants(alpha_Qcc)

        if self.observation['geometry'] == '-Z(XX)Z':  # Porto's notation
            return (45 * alpha2_r + 5 * delta2_r + 4 * gamma2_r) / 45.
        elif self.observation['geometry'] == '-Z(XY)Z':  # Porto's notation
            return gamma2_r / 15.
        elif self.observation['scattered'] == 'Z':
            # scattered light in direction of incoming light
            return (45 * alpha2_r + 5 * delta2_r + 7 * gamma2_r) / 45.
        elif self.observation['scattered'] == 'parallel':
            # scattered light perendicular and
            # polarization in plane
            return 6 * gamma2_r / 45.
        elif self.observation['scattered'] == 'perpendicular':
            # scattered light perendicular and
            # polarization out of plane
            return (45 * alpha2_r + 5 * delta2_r + 7 * gamma2_r) / 45.
        else:
            raise NotImplementedError

    def _invariants(self, alpha_Qcc):
        """Raman invariants

        Parameter
        ---------
        alpha_Qcc: array
           Matrix element or polarizability tensor

        Reference
        ---------
        Derek A. Long, The Raman Effect, ISBN 0-471-49028-8

        Returns
        -------
        mean polarizability, anisotropy, asymmetric anisotropy
        """
        m2 = Raman.m2
        alpha2_r = m2(alpha_Qcc[:, 0, 0] + alpha_Qcc[:, 1, 1] +
                      alpha_Qcc[:, 2, 2]) / 9.
        delta2_r = 3 / 4. * (
            m2(alpha_Qcc[:, 0, 1] - alpha_Qcc[:, 1, 0]) +
            m2(alpha_Qcc[:, 0, 2] - alpha_Qcc[:, 2, 0]) +
            m2(alpha_Qcc[:, 1, 2] - alpha_Qcc[:, 2, 1]))
        gamma2_r = (3 / 4. * (m2(alpha_Qcc[:, 0, 1] + alpha_Qcc[:, 1, 0]) +
                              m2(alpha_Qcc[:, 0, 2] + alpha_Qcc[:, 2, 0]) +
                              m2(alpha_Qcc[:, 1, 2] + alpha_Qcc[:, 2, 1])) +
                    (m2(alpha_Qcc[:, 0, 0] - alpha_Qcc[:, 1, 1]) +
                     m2(alpha_Qcc[:, 0, 0] - alpha_Qcc[:, 2, 2]) +
                     m2(alpha_Qcc[:, 1, 1] - alpha_Qcc[:, 2, 2])) / 2)
        return alpha2_r, gamma2_r, delta2_r

    def summary(self,
                method='standard', direction='central',
                log=sys.stdout):
        """Print summary for given omega [eV]"""
        self.read(method, direction)
        hnu = self.get_energies()
        intensities = self.get_absolute_intensities()
        te = int(np.log10(intensities.max())) - 2
        scale = 10**(-te)
        if not te:
            ts = ''
        elif te > -2 and te < 3:
            ts = str(10**te)
        else:
            ts = '10^{0}'.format(te)

        if isinstance(log, str):
            log = paropen(log, 'a')

        parprint('-------------------------------------', file=log)
        parprint(' Mode    Frequency        Intensity', file=log)
        parprint('  #    meV     cm^-1      [{0}A^4/amu]'.format(ts), file=log)
        parprint('-------------------------------------', file=log)
        for n, e in enumerate(hnu):
            if e.imag != 0:
                c = 'i'
                e = e.imag
            else:
                c = ' '
                e = e.real
            parprint('%3d %6.1f%s  %7.1f%s  %9.2f' %
                     (n, 1000 * e, c, e / u.invcm, c, intensities[n] * scale),
                     file=log)
        parprint('-------------------------------------', file=log)
        parprint('Zero-point energy: %.3f eV' %
                 self.vibrations.get_zero_point_energy(),
                 file=log)


class Raman(RamanEvaluate, Vibrations):
    def read_energies_and_modes(self, *args, **kwargs):
        self.timer.start('vibrations')
        self.vibrations.read(*args, **kwargs)
        self.timer.stop('vibrations')

        # XXX move this to another place
        self.im_r = self.vibrations.im
        self.modes = self.vibrations.modes
        self.om_Q = self.vibrations.hnu.real    # energies in eV
        self.H = self.vibrations.H   # XXX used in albrecht.py
        # pre-factors for one vibrational excitation
        with np.errstate(divide='ignore'):
            self.vib01_Q = np.where(self.om_Q > 0,
                                    1. / np.sqrt(2 * self.om_Q), 0)
        # -> sqrt(amu) * Angstrom
        self.vib01_Q *= np.sqrt(u.Ha * u._me / u._amu) * u.Bohr


class RamanPhonons(RamanEvaluate, Phonons):
    def read_energies_and_modes(self, method='standard', direction='central'):
        self.timer.start('band_structure')
        Phonons.read(self, method, direction)
        
        kpts_kc = monkhorst_pack(self.kpts)
        omega_kl, u_kl = self.band_structure(kpts_kc, modes=True,
                                             verbose=self.verbose)

        self.im = self.m_inv_x  # use the same name as in Vibrations
        
        # we now have:
        # self.H     : Hessian matrix
        # self.im    : 1./sqrt(masses)
        # self.modes : Eigenmodes of the mass weighted Hessian
        self.om_Q = self.hnu.real    # energies in eV
        self.om_v = self.om_Q
        # pre-factors for one vibrational excitation
        with np.errstate(divide='ignore'):
            self.vib01_Q = np.where(self.om_Q > 0,
                                    1. / np.sqrt(2 * self.om_Q), 0)
        # -> sqrt(amu) * Angstrom
        self.vib01_Q *= np.sqrt(u.Ha * u._me / u._amu) * u.Bohr
        self.timer.stop('band_structure')
