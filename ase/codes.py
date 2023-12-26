from dataclasses import dataclass

# requires external program
# 

@dataclass
class CodeMetadata:
    codes = {}

    name: str
    longname: str
    modulename: str
    classname: str

    def calculator_class(self):
        from importlib import import_module
        module = import_module(self.modulename)
        cls = getattr(module, self.classname)
        # assert cls.name == self.name, f'{cls.name} vs {self.name}'
        return cls

    @classmethod
    def register(cls, name, longname, importpath):
        modulename, classname = importpath.rsplit('.', 1)
        code = cls(name, longname, modulename, classname)
        cls.codes[name] = code
        return code

    def _description(self):
        yield self.name
        yield f'Name:     {self.longname}'
        yield f'Location: {self.modulename}.{self.classname}'
        yield f'Type:     {self.calculator_type()}'

    def description(self):
        tokens = [*self._description()]
        tokens[1:] = [f'  {token}' for token in tokens[1:]]
        return '\n'.join(tokens)

    def is_legacy_fileio(self):
        from ase.calculators.calculator import FileIOCalculator
        return issubclass(self.calculator_class(), FileIOCalculator)

    def is_generic_fileio(self):
        from ase.calculators.genericfileio import GenericFileIOCalculator
        return issubclass(self.calculator_class(), GenericFileIOCalculator)

    def is_calculator_oldbase(self):
        from ase.calculators.calculator import Calculator
        return issubclass(self.calculator_class(), Calculator)

    def is_base_calculator(self):
        from ase.calculators.calculator import BaseCalculator
        return issubclass(self.calculator_class(), BaseCalculator)

    def calculator_type(self):

        if self.is_generic_fileio():
            return 'GenericFileIOCalculator'

        if self.is_legacy_fileio():
            return 'FileIOCalculator (legacy)'

        if self.is_calculator_oldbase():
            return 'Calculator (legacy base class)'

        if self.is_base_calculator:
            return 'Base calculator'

        mro = self.calculator_class().__mro__
        return f'BAD: Not a proper calculator (superclasses: {mro})'


R = CodeMetadata.register


R('abinit', 'Abinit', 'ase.calculators.abinit.Abinit')
R('ace', 'ACE molecule', 'ase.calculators.acemolecule.ACE')
# internal: R('acn', 'ACN force field', 'ase.calculators.acn.ACN')
R('aims', 'FHI-Aims', 'ase.calculators.aims.Aims')
R('amber', 'Amber', 'ase.calculators.amber.Amber')
R('castep', 'Castep', 'ase.calculators.castep.Castep')
# internal: combine_mm
# internal: counterions
R('cp2k', 'CP2K', 'ase.calculators.cp2k.CP2K')
R('crystal', 'Crystap', 'ase.calculators.crystal.CRYSTAL')
R('demon', 'deMon', 'ase.calculators.demon.Demon')
R('demonnano', 'deMon-nano', 'ase.calculators.demonnano.DemonNano')
R('dftb', 'DFTB+', 'ase.calculators.dftb.Dftb')
R('dftd3', 'DFT-D3', 'ase.calculators.dftd3.DFTD3')
# R('dftd3-pure', 'DFT-D3 (pure)', 'ase.calculators.dftd3.puredftd3')
R('dmol', 'DMol3', 'ase.calculators.dmol.DMol3')
# internal: R('eam', 'EAM', 'ase.calculators.eam.EAM')
R('elk', 'ELK', 'ase.calculators.elk.ELK')
# internal: R('emt', 'EMT potential', 'ase.calculators.emt.EMT')
R('espresso', 'Quantum Espresso', 'ase.calculators.espresso.Espresso')
R('exciting', 'Exciting',
  'ase.calculators.exciting.exciting.ExcitingGroundStateCalculator')
R('ff', 'FF', 'ase.calculators.ff.ForceField')
# fleur <- external nowadays
R('gamess_us', 'GAMESS-US', 'ase.calculators.gamess_us.GAMESSUS')
R('gaussian', 'Gaussian', 'ase.calculators.gaussian.Gaussian')
R('gromacs', 'Gromacs', 'ase.calculators.gromacs.Gromacs')
R('gulp', 'GULP', 'ase.calculators.gulp.GULP')
# h2morse.py really?
# internal: R('harmonic', 'Harmonic potential',
#  'ase.calculators.harmonic.HarmonicCalculator')
# internal: R('idealgas', 'Ideal gas (dummy)',
#             'ase.calculators.idealgas.IdealGas')
# XXX cannot import without kimpy installed, fixme:
# R('kim', 'OpenKIM', 'ase.calculators.kim.kim.KIM')
R('lammpslib', 'Lammps (python library)', 'ase.calculators.lammpslib.LAMMPSlib')
R('lammpsrun', 'Lammps (external)', 'ase.calculators.lammpsrun.LAMMPS')
# internal: R('lj', 'Lennard–Jones potential',
#             'ase.calculators.lj.LennardJones')
# internal: loggingcalc.py
# internal: mixing.py
R('mopac', 'MOPAC', 'ase.calculators.mopac.MOPAC')
# internal: R('morse', 'Morse potential',
# 'ase.calculators.morse.MorsePotential')
R('nwchem', 'NWChem', 'ase.calculators.nwchem.NWChem')
R('octopus', 'Octopus', 'ase.calculators.octopus.Octopus')
R('onetep', 'Onetep', 'ase.calculators.onetep.Onetep')
R('openmx', 'OpenMX', 'ase.calculators.openmx.OpenMX')
R('orca', 'ORCA', 'ase.calculators.orca.ORCA')
R('plumed', 'Plumed', 'ase.calculators.plumed.Plumed')
R('psi4', 'Psi4', 'ase.calculators.psi4.Psi4')
R('qchem', 'QChem', 'ase.calculators.qchem.QChem')
# internal: qmmm.py
R('siesta', 'SIESTA', 'ase.calculators.siesta.Siesta')
# internal: test.py
# internal: R('tip3p', 'TIP3P', 'ase.calculators.tip3p.TIP3P')
# internal: R('tip4p', 'TIP4P', 'ase.calculators.tip4p.TIP4P')
R('turbomole', 'Turbomole', 'ase.calculators.turbomole.Turbomole')
R('vasp', 'VASP', 'ase.calculators.vasp.Vasp')
# internal: vdwcorrection


def list_codes():
    for code in CodeMetadata.codes.values():
        try:
            print(code.description())
        except Exception as ex:
            raise RuntimeError(code) from ex
        print()


if __name__ == '__main__':
    list_codes()
