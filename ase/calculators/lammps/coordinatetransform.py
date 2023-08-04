"""Prism"""
import warnings

import numpy as np
from ase.geometry import wrap_positions

# Order in which off-diagonal elements are checked for strong tilt
FLIP_ORDER = [(1, 0, 0), (2, 0, 0), (2, 1, 1)]


def calc_box_parameters(cell: np.ndarray) -> np.ndarray:
    """Calculate box parameters

    https://docs.lammps.org/Howto_triclinic.html
    """
    ax = np.sqrt(cell[0] @ cell[0])
    bx = cell[0] @ cell[1] / ax
    by = np.sqrt(cell[1] @ cell[1] - bx ** 2)
    cx = cell[0] @ cell[2] / ax
    cy = (cell[1] @ cell[2] - bx * cx) / by
    cz = np.sqrt(cell[2] @ cell[2] - cx ** 2 - cy ** 2)
    return np.array((ax, by, cz, bx, cx, cy))


def calc_lammps_tilt(cell: np.ndarray) -> np.ndarray:
    """Calculate rotated cell in LAMMPS coordinates"""
    ax, by, cz, bx, cx, cy = calc_box_parameters(cell)
    return np.array(((ax, 0.0, 0.0), (bx, by, 0.0), (cx, cy, cz)))


class Prism:
    """The representation of the unit cell in LAMMPS

    The main purpose of the prism-object is to create suitable
    string representations of prism limits and atom positions
    within the prism.

    Parameters
    ----------
    cell : np.ndarray
        Cell in ASE coordinate system.
    pbc : one or three bool
        Periodic boundary conditions flags.
    tolerance : float
        Precision for skewness test.

    Notes
    -----
    LAMMPS requires triangular matrixes without a strong tilt.
    Therefore the 'Prism'-object contains three coordinate systems:

    - ase_cell (the simulated system in the ASE coordination system)
    - lammps_tilt (ase-cell rotated to be an lower triangular matrix)
    - lammps_cell (same volume as tilted cell, but reduce edge length)

    The translation between 'ase_cell' and 'lammps_cell' is down with a
    rotation matrix 'rot_mat' obtained from a QR decomposition.  The
    transformation between 'lammps_tilt' and 'lammps_cell' is done by
    changing the off-diagonal elements.  'flip' saves the modified
    elements for later reversal.  All vectors except positions are just
    rotated with 'rot_mat'.  Positions are rotated and wrapped into
    'lammps_cell'.  Translating results back from LAMMPS needs first the
    unwrapping of positions.  Then all vectors and the unit-cell are
    rotated back into the ASE coordination system.  This can fail as
    depending on the simulation run LAMMPS might have changed the
    simulation box significantly.  This is for example a problem with
    hexagonal cells.  LAMMPS might also wrap atoms across periodic
    boundaries, which can lead to problems for example NEB
    calculations.
    """

    # !TODO: derive tolerance from cell-dimensions
    def __init__(self, cell, pbc=True, tolerance=1.0e-8):
        # Use QR decomposition to get the lammps cell
        #    rot_mat * lammps_tilt^T = ase_cell^T
        # => lammps_tilt * rot_mat^T = ase_cell
        # => lammps_tilt             = ase_cell * rot_mat
        # LAMMPS requires positive diagonal elements of the triangular matrix.
        # The diagonals of `lammps_tilt` are always positive by construction.
        self.lammps_tilt = calc_lammps_tilt(cell)
        self.rot_mat = np.linalg.solve(self.lammps_tilt, cell).T
        self.ase_cell = cell
        self.tolerance = tolerance
        self.pbc = np.zeros(3, bool) + pbc
        self.flip = np.zeros(3, bool)
        self.lammps_cell = self.calc_lammps_cell()

    def calc_lammps_cell(self) -> np.ndarray:
        """Calculate LAMMPS cell with short lattice basis vectors"""
        # LAMMPS minimizes the edge length of the parallelepiped
        # What is ment with 'flip': cell 2 is transformed into cell 1
        # cell 2 = 'lammps_tilt'; cell 1 = 'lammps_cell'
        # o-----------------------------/==o-----------------------------/--o
        #  \                        /--/    \                        /--/
        #   \                   /--/         \                   /--/
        #    \         1    /--/              \   2          /--/
        #     \         /--/                   \         /--/
        #      \    /--/                        \    /--/
        #       o==/-----------------------------o--/
        # !TODO: handle extreme tilt (= off-diagonal > 1.5)
        lammps_cell = self.lammps_tilt.copy()
        for iteri, (i, j, k) in enumerate(FLIP_ORDER):
            if not self.pbc[k]:
                continue
            if abs(lammps_cell[i][j] / self.lammps_tilt[k][k]) > 0.5:
                self.flip[iteri] = True
                change = lammps_cell[k][k]
                change *= np.sign(lammps_cell[i][j])
                lammps_cell[i][j] -= change
        return lammps_cell

    def get_lammps_prism(self) -> np.ndarray:
        """Return box parameters of the rotated cell in LAMMPS coordinates

        Returns
        -------
        np.ndarray
            xhi - xlo, yhi - ylo, zhi - zlo, xy, xz, yz
        """
        return self.lammps_cell[(0, 1, 2, 1, 2, 2), (0, 1, 2, 0, 0, 1)]

    # !TODO: detect flip in lammps
    def update_cell(self, lammps_cell: np.ndarray) -> np.ndarray:
        """Rotate new LAMMPS cell into ASE coordinate system

        Parameters
        ----------
        lammps_cell : np.ndarray
            New Cell in LAMMPS coordinates received after executing LAMMPS

        Returns
        -------
        np.ndarray
            New cell in ASE coordinates
        """
        self.lammps_cell = lammps_cell
        self.lammps_tilt = self.lammps_cell.copy()

        # reverse flip
        for iteri, (i, j, k) in enumerate(FLIP_ORDER):
            if self.flip[iteri]:
                change = self.lammps_cell[k][k]
                change *= np.sign(self.lammps_cell[i][j])
                self.lammps_tilt[i][j] -= change

        # try to detect potential flips in lammps
        # (lammps minimizes the cell-vector lengths)
        new_ase_cell = np.dot(self.lammps_tilt, self.rot_mat.T)
        # assuming the cell changes are mostly isotropic
        new_vol = np.linalg.det(new_ase_cell)
        old_vol = np.linalg.det(self.ase_cell)
        test_residual = self.ase_cell.copy()
        test_residual *= (new_vol / old_vol) ** (1.0 / 3.0)
        test_residual -= new_ase_cell
        if any(
                np.linalg.norm(test_residual, axis=1)
                > 0.5 * np.linalg.norm(self.ase_cell, axis=1)
        ):
            warnings.warn(
                "Significant simulation cell changes from LAMMPS detected. "
                "Backtransformation to ASE might fail!"
            )
        return new_ase_cell

    def vector_to_lammps(
        self,
        vec: np.ndarray,
        wrap: bool = False,
    ) -> np.ndarray:
        """Rotate vectors from ASE to LAMMPS coordinates

        Parameters
        ----------
        vec : np.ndarray
            Vectors in ASE coordinates to be rotated into LAMMPS coordinates
        wrap : bool
            If True, the vectors are wrapped into the cell

        Returns
        -------
        np.array
            Vectors in LAMMPS coordinates
        """
        # !TODO: right eps-limit
        # lammps might not like atoms outside the cell
        if wrap:
            return wrap_positions(
                np.dot(vec, self.rot_mat),
                self.lammps_cell,
                pbc=self.pbc,
                eps=1e-18,
            )
        return np.dot(vec, self.rot_mat)

    def vector_to_ase(
        self,
        vec: np.ndarray,
        wrap: bool = False,
    ) -> np.ndarray:
        """Rotate vectors from LAMMPS to ASE coordinates

        Parameters
        ----------
        vec : np.ndarray
            Vectors in LAMMPS coordinates to be rotated into ASE coordinates
        wrap : bool
            If True, the vectors are wrapped into the cell

        Returns
        -------
        np.ndarray
            Vectors in ASE coordinates
        """
        if wrap:
            fractional = np.linalg.solve(self.lammps_tilt.T, vec.T).T
            # wrap into 0 to 1 for periodic directions
            fractional -= np.floor(fractional) * self.pbc
            vec = np.dot(fractional, self.lammps_tilt)
        return np.dot(vec, self.rot_mat.T)

    def is_skewed(self) -> bool:
        """Test if the lammps cell is skewed, i.e., monoclinic or triclinic.

        Returns
        -------
        bool
            True if the lammps cell is skewed.
        """
        cell_sq = self.lammps_cell ** 2
        on_diag = np.sum(np.diag(cell_sq))
        off_diag = np.sum(np.tril(cell_sq, -1))
        return off_diag / on_diag > self.tolerance
