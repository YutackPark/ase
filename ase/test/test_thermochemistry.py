import os
from shutil import rmtree

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


def teardown_module():
    for f in os.listdir(os.getcwd()):
        if "-vib" in f:
            rmtree(f)


def test_ideal_gas_thermo(testdir):
    # TEST 1: We do a basic test on N2
    atoms = Atoms("N2", positions=[(0, 0, 0), (0, 0, 1.1)])
    atoms.calc = EMT()
    QuasiNewton(atoms).run(fmax=0.01)
    energy = atoms.get_potential_energy()
    vib = Vibrations(atoms, name="igt-vib1")
    vib.run()
    vib_energies = vib.get_energies()
    assert len(vib_energies) == 6
    assert vib_energies[0] == pytest.approx(0.0)
    assert vib_energies[-1] == pytest.approx(1.52647479e-01)

    # ---------------------
    #   #    meV     cm^-1
    # ---------------------
    #   0    0.0       0.0 <--- remove!
    #   1    0.0       0.0 <--- remove!
    #   2    0.0       0.0 <--- remove!
    #   3    1.7      13.5 <--- remove!
    #   4    1.7      13.5 <--- remove!
    #   5  152.6    1231.2
    # ---------------------
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

    # TEST 2: Now we try something a bit harder. Let's consider a
    # CH3 molecule, such that there should be 3*4-6 = 6 modes
    # for calculating the thermochemistry. We will also provide
    # the modes in an unsorted list to make sure the correct
    # values are cut. Note that these vibrational energies
    # are simply toy values.

    # Input: [1.0, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.35, 0.12]
    # Expected: [0.12, 0.2, 0.3, 0.35, 0.4, 1.0]
    thermo = IdealGasThermo(
        vib_energies=[1.0, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.35, 0.12],
        geometry="nonlinear",
        atoms=molecule("CH3"),
        symmetrynumber=6,
        potentialenergy=9,
        spin=0.5,
    )
    assert len(thermo.vib_energies) == 6
    assert list(thermo.vib_energies) == [0.12, 0.2, 0.3, 0.35, 0.4, 1.0]
    assert thermo.atoms == molecule("CH3")
    assert thermo.geometry == "nonlinear"
    assert thermo.get_ZPE_correction() == pytest.approx(1.185)
    assert thermo.get_enthalpy(1000) == pytest.approx(10.610695269124156)
    assert thermo.get_entropy(1000, 1e8) == pytest.approx(0.0019310086280219891)
    assert thermo.get_gibbs_energy(1000, 1e8) == pytest.approx(
        thermo.get_enthalpy(1000) - 1000 * thermo.get_entropy(1000, 1e8)
    )

    # TEST 3: Now we give the module a more complicated set of
    # vibrational frequencies to deal with to make sure
    # the correct values are cut. This structure is not a
    # minimum or TS and has several imaginary modes. However
    # if we just the first 6 modes, it'd look like all are
    # real when they are not. We need to cut based on
    # np.abs() of the vibrational energies.

    # ---------------------
    #   #    meV     cm^-1
    # ---------------------
    #   0   63.8i    514.8i
    #   1   63.3i    510.7i
    #   2   42.4i    342.3i
    #   3    5.3i     43.1i <--- remove!
    #   4    0.0       0.0  <--- remove!
    #   5    0.0       0.0  <--- remove!
    #   6    0.0       0.0  <--- remove!
    #   7    5.6      45.5  <--- remove!
    #   8    6.0      48.1  <--- remove!
    #   9  507.9    4096.1
    #  10  547.2    4413.8
    #  11  547.7    4417.3
    # ---------------------
    vib_energies = [
        63.8j,
        63.3j,
        42.4j,
        5.3j,
        0.0,
        0.0,
        0.0,
        5.6,
        6.0,
        507.9,
        547.2,
        547.7,
    ]
    with pytest.raises(ValueError):
        # Imaginary frequencies present!!!
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            geometry="nonlinear",
            atoms=molecule("CH3"),
            symmetrynumber=6,
            potentialenergy=0.0,
            spin=0.5,
        )

    # TEST 4: Let's do another test like above, just for fun.
    # Again, this is not a minimum or TS and has several
    # imaginary modes.
    atoms = molecule("CH3")
    atoms.calc = EMT()
    vib = Vibrations(atoms, name="igt-vib2")
    vib.run()
    vib_energies = vib.get_energies()
    assert len(vib_energies) == 12
    assert vib_energies[0] == pytest.approx(0.09599611291404943j)
    assert vib_energies[-1] == pytest.approx(0.39035516516367375)

    # ---------------------
    #   #    meV     cm^-1
    # ---------------------
    #   0   96.0i    774.3i
    #   1   89.4i    721.0i
    #   2   89.3i    720.4i
    #   3   85.5i    689.7i <-- remove!
    #   4   85.4i    689.1i <-- remove!
    #   5   85.4i    689.1i <-- remove!
    #   6    0.0       0.0 <-- remove!
    #   7    0.0       0.0 <-- remove!
    #   8    0.0       0.0 <-- remove!
    #   9  369.4    2979.1
    #  10  369.4    2979.3
    #  11  390.4    3148.4
    # ---------------------
    with pytest.raises(ValueError):
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            geometry="nonlinear",
            atoms=atoms,
            symmetrynumber=6,
            potentialenergy=0.0,
            spin=0.5,
        )

    # TEST 5: Do a sanity check if given nonsensical vibrational energies
    # with real and imag parts
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
        name="harmonic-vib1",
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
    assert len(thermo.vib_energies) == len(vib_energies) - 3
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
    assert len(thermo.vib_energies) == len(vib_energies) - 3
    assert helmholtz == pytest.approx(1.5932242071261076)

    atoms = bulk("Cu") * (2, 2, 2)
    thermo = HinderedThermo(
        vib_energies=vib_energies,
        trans_barrier_energy=trans_barrier_energy,
        rot_barrier_energy=rot_barrier_energy,
        sitedensity=sitedensity,
        rotationalminima=rotationalminima,
        symmetrynumber=symmetrynumber,
        mass=mass,
        inertia=inertia,
        atoms=atoms,
    )
    assert len(thermo.vib_energies) == 3 * len(atoms) - 3

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
