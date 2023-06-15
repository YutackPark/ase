import numpy as np
import pytest

from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc100, molecule
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.phonons import Phonons
from ase.thermochemistry import (
    CrystalThermo,
    HarmonicThermo,
    HinderedThermo,
    IdealGasThermo,
)
from ase.vibrations import Vibrations


def test_ideal_gas_thermo(testdir):
    atoms = Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.1)])
    atoms.calc = EMT()
    QuasiNewton(atoms).run(fmax=0.01)
    energy = atoms.get_potential_energy()
    vib = Vibrations(atoms, name="idealgasthermo-vib")
    vib.run()
    vib_energies = vib.get_energies()
    assert len(vib_energies) == 6
    assert vib_energies[0] == pytest.approx(0.0)
    assert vib_energies[-1] == pytest.approx(1.52647479e-01)

    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        geometry="linear",
        atoms=atoms,
        symmetrynumber=2,
        spin=0,
        potentialenergy=energy,
    )
    assert len(thermo.vib_energies) == 1
    assert thermo.vib_energies[0] == vib_energies[-1]
    assert thermo.atoms == atoms
    assert thermo.geometry == "linear"
    assert thermo.get_ZPE_correction() == pytest.approx(0.07632373926263808)
    assert thermo.get_enthalpy(1000) == pytest.approx(0.6719935644272014)
    assert thermo.get_entropy(1000, 1e8) == pytest.approx(0.0017861226676818658)
    assert thermo.get_gibbs_energy(1000, 1e8) == pytest.approx(
        thermo.get_enthalpy(1000) - 1000 * thermo.get_entropy(1000, 1e8)
    )

    atoms = molecule("CH3")
    atoms.calc = EMT()
    QuasiNewton(atoms).run(fmax=0.01)
    energy = atoms.get_potential_energy()
    vib = Vibrations(atoms, name="idealgasthermo-vib2")
    vib.run()
    vib_energies = vib.get_energies()
    assert len(vib_energies) == 12
    assert vib_energies[0] == pytest.approx(0.013170007749561785j)
    assert vib_energies[-1] == pytest.approx(0.3323345144618613)
    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        geometry="nonlinear",
        atoms=atoms,
        symmetrynumber=6,
        potentialenergy=energy,
        spin=0.5,
    )
    assert len(thermo.vib_energies) == 6
    assert thermo.vib_energies[0] == vib_energies[6]
    assert thermo.vib_energies[-1] == vib_energies[-1]
    assert thermo.atoms == atoms
    assert thermo.geometry == "nonlinear"
    assert thermo.get_ZPE_correction() == pytest.approx(0.4794163027968802)
    assert thermo.get_enthalpy(1000) == pytest.approx(2.912700987111111)
    assert thermo.get_entropy(1000, 1e8) == pytest.approx(0.0029295583320900396)
    assert thermo.get_gibbs_energy(1000, 1e8) == pytest.approx(
        thermo.get_enthalpy(1000) - 1000 * thermo.get_entropy(1000, 1e8)
    )

    thermo = IdealGasThermo(
        vib_energies=vib_energies[-12:],
        geometry="nonlinear",
        atoms=atoms,
        symmetrynumber=6,
        potentialenergy=energy,
        spin=0.5,
    )
    assert len(thermo.vib_energies) == 6
    assert thermo.atoms == atoms
    assert thermo.geometry == "nonlinear"
    assert thermo.get_ZPE_correction() == pytest.approx(0.4794163027968802)
    assert thermo.get_enthalpy(1000) == pytest.approx(2.912700987111111)
    assert thermo.get_entropy(1000, 1e8) == pytest.approx(0.0029295583320900396)
    assert thermo.get_gibbs_energy(1000, 1e8) == pytest.approx(
        thermo.get_enthalpy(1000) - 1000 * thermo.get_entropy(1000, 1e8)
    )

    vib_energies = list(vib_energies)
    vib_energies.sort(reverse=True)
    thermo = IdealGasThermo(
        vib_energies=vib_energies,
        geometry="nonlinear",
        atoms=atoms,
        symmetrynumber=6,
        potentialenergy=energy,
        spin=0.5,
    )
    assert len(thermo.vib_energies) == 6
    assert thermo.atoms == atoms
    assert thermo.geometry == "nonlinear"
    assert thermo.get_ZPE_correction() == pytest.approx(0.4794163027968802)
    assert thermo.get_enthalpy(1000) == pytest.approx(2.912700987111111)
    assert thermo.get_entropy(1000, 1e8) == pytest.approx(0.0029295583320900396)
    assert thermo.get_gibbs_energy(1000, 1e8) == pytest.approx(
        thermo.get_enthalpy(1000) - 1000 * thermo.get_entropy(1000, 1e8)
    )

    with pytest.raises(ValueError):
        thermo = IdealGasThermo(
            vib_energies=[100 + 0.1j] * len(vib_energies),
            geometry="nonlinear",
            atoms=atoms,
            symmetrynumber=6,
            potentialenergy=energy,
            spin=0.5,
        )


def test_harmonic_thermo(testdir):
    atoms = fcc100("Cu", (2, 2, 2), vacuum=10.0)
    atoms.calc = EMT()
    add_adsorbate(atoms, "Pt", 1.5, "hollow")
    atoms.set_constraint(
        FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == "Cu"])
    )
    QuasiNewton(atoms).run(fmax=0.01)
    vib = Vibrations(
        atoms,
        name="harmonicthermo-vib",
        indices=[atom.index for atom in atoms if atom.symbol != "Cu"],
    )
    vib.run()
    vib.summary()
    vib_energies = vib.get_energies()

    thermo = HarmonicThermo(
        vib_energies=vib_energies, potentialenergy=atoms.get_potential_energy()
    )
    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert helmholtz == pytest.approx(4.060698673180732)

    vib_energies = list(vib_energies)
    vib_energies.sort(reverse=True)
    thermo = HarmonicThermo(
        vib_energies=vib_energies, potentialenergy=atoms.get_potential_energy()
    )
    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert helmholtz == pytest.approx(4.060698673180732)


def test_crystal_thermo(asap3, testdir):
    atoms = bulk("Al", "fcc", a=4.05)
    calc = asap3.EMT()
    atoms.calc = calc
    energy = atoms.get_potential_energy()

    # Phonon calculator
    N = 7
    ph = Phonons(atoms, calc, supercell=(N, N, N), delta=0.05)
    ph.run()

    ph.read(acoustic=True)
    phonon_energies, phonon_DOS = ph.dos(kpts=(4, 4, 4), npts=30, delta=5e-4)

    thermo = CrystalThermo(
        phonon_energies=phonon_energies,
        phonon_DOS=phonon_DOS,
        potentialenergy=energy,
        formula_units=4,
    )
    thermo.get_helmholtz_energy(temperature=298.15)


def test_hindered_thermo():
    # Hindered translator / rotor.
    # (Taken directly from the example given in the documentation.)

    vibs = np.array(
        [
            3049.060670,
            3040.796863,
            3001.661338,
            2997.961647,
            2866.153162,
            2750.855460,
            1436.792655,
            1431.413595,
            1415.952186,
            1395.726300,
            1358.412432,
            1335.922737,
            1167.009954,
            1142.126116,
            1013.918680,
            803.400098,
            783.026031,
            310.448278,
            136.112935,
            112.939853,
            103.926392,
            77.262869,
            60.278004,
            25.825447,
        ]
    )
    vib_energies = vibs / 8065.54429  # Convert to eV from cm^-1.
    trans_barrier_energy = 0.049313  # eV
    rot_barrier_energy = 0.017675  # eV
    sitedensity = 1.5e15  # cm^-2
    rotationalminima = 6
    symmetrynumber = 1
    mass = 30.07  # amu
    inertia = 73.149  # amu Ang^-2

    thermo = HinderedThermo(
        vib_energies=vib_energies,
        trans_barrier_energy=trans_barrier_energy,
        rot_barrier_energy=rot_barrier_energy,
        sitedensity=sitedensity,
        rotationalminima=rotationalminima,
        symmetrynumber=symmetrynumber,
        mass=mass,
        inertia=inertia,
    )
    assert len(thermo.vib_energies) == len(vib_energies)-3
    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert helmholtz == pytest.approx(1.5932242071261076)

    vib_energies = list(vib_energies)
    vib_energies.sort(reverse=True)
    thermo = HinderedThermo(
        vib_energies=vib_energies,
        trans_barrier_energy=trans_barrier_energy,
        rot_barrier_energy=rot_barrier_energy,
        sitedensity=sitedensity,
        rotationalminima=rotationalminima,
        symmetrynumber=symmetrynumber,
        mass=mass,
        inertia=inertia,
    )

    helmholtz = thermo.get_helmholtz_energy(temperature=298.15)
    assert len(thermo.vib_energies) == len(vib_energies)-3
    assert helmholtz == pytest.approx(1.5932242071261076)

    atoms = bulk("Cu")*(2, 2, 2)
    thermo = HinderedThermo(
        vib_energies=vib_energies,
        trans_barrier_energy=trans_barrier_energy,
        rot_barrier_energy=rot_barrier_energy,
        sitedensity=sitedensity,
        rotationalminima=rotationalminima,
        symmetrynumber=symmetrynumber,
        mass=mass,
        inertia=inertia,
        atoms=atoms
    )
    assert len(thermo.vib_energies) == 3*len(atoms)-3

    with pytest.raises(ValueError):
        thermo = HinderedThermo(
            vib_energies=[100 + 0.1j] * len(vib_energies),
            trans_barrier_energy=trans_barrier_energy,
            rot_barrier_energy=rot_barrier_energy,
            sitedensity=sitedensity,
            rotationalminima=rotationalminima,
            symmetrynumber=symmetrynumber,
            mass=mass,
            inertia=inertia,
        )
