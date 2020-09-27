"""
Module for povray file format support.

See http://www.povray.org/ for details on the format.
"""
from typing import Dict, Any

import numpy as np

from ase.io.utils import PlottingVariables
from ase.constraints import FixAtoms
from ase import Atoms
from subprocess import check_call, DEVNULL
from pathlib import Path


def pa(array):
    """Povray array syntax"""
    return '<% 6.2f, % 6.2f, % 6.2f>' % tuple(array)


def pc(array):
    """Povray color syntax"""
    if isinstance(array, str):
        return 'color ' + array
    if isinstance(array, float):
        return 'rgb <%.2f>*3' % array
    if len(array) == 3:
        return 'rgb <%.2f, %.2f, %.2f>' % tuple(array)
    if len(array) == 4:  # filter
        return 'rgbt <%.2f, %.2f, %.2f, %.2f>' % tuple(array)
    if len(array) == 5:  # filter and transmit
        return 'rgbft <%.2f, %.2f, %.2f, %.2f, %.2f>' % tuple(array)


def get_bondpairs(atoms, radius=1.1):
    """Get all pairs of bonding atoms

    Return all pairs of atoms which are closer than radius times the
    sum of their respective covalent radii.  The pairs are returned as
    tuples::

      (a, b, (i1, i2, i3))

    so that atoms a bonds to atom b displaced by the vector::

        _     _     _
      i c + i c + i c ,
       1 1   2 2   3 3

    where c1, c2 and c3 are the unit cell vectors and i1, i2, i3 are
    integers."""

    from ase.data import covalent_radii
    from ase.neighborlist import NeighborList
    cutoffs = radius * covalent_radii[atoms.numbers]
    nl = NeighborList(cutoffs=cutoffs, self_interaction=False)
    nl.update(atoms)
    bondpairs = []
    for a in range(len(atoms)):
        indices, offsets = nl.get_neighbors(a)
        bondpairs.extend([(a, a2, offset)
                          for a2, offset in zip(indices, offsets)])
    return bondpairs


def set_high_bondorder_pairs(bondpairs, high_bondorder_pairs=None):
    """Set high bondorder pairs

    Modify bondpairs list (from get_bondpairs((atoms)) to include high
    bondorder pairs.

    Parameters:
    -----------
    bondpairs: List of pairs, generated from get_bondpairs(atoms)
    high_bondorder_pairs: Dictionary of pairs with high bond orders
                          using the following format:
                          { ( a1, b1 ): ( offset1, bond_order1, bond_offset1),
                            ( a2, b2 ): ( offset2, bond_order2, bond_offset2),
                            ...
                          }
                          offset, bond_order, bond_offset are optional.
                          However, if they are provided, the 1st value is
                          offset, 2nd value is bond_order,
                          3rd value is bond_offset """

    if high_bondorder_pairs is None:
        high_bondorder_pairs = dict()
    bondpairs_ = []
    for pair in bondpairs:
        (a, b) = (pair[0], pair[1])
        if (a, b) in high_bondorder_pairs.keys():
            bondpair = [a, b] + [item for item in high_bondorder_pairs[(a, b)]]
            bondpairs_.append(bondpair)
        elif (b, a) in high_bondorder_pairs.keys():
            bondpair = [a, b] + [item for item in high_bondorder_pairs[(b, a)]]
            bondpairs_.append(bondpair)
        else:
            bondpairs_.append(pair)
    return bondpairs_


class POVRAY(PlottingVariables):
    default_settings: Dict[str, Any] = {
        # x, y is the image plane, z is *out* of the screen
        'display': False,  # display while rendering
        'pause': True,  # pause when done rendering (only if display)
        'transparent': True,  # transparent background
        'canvas_width': None,  # width of canvas in pixels
        'canvas_height': None,  # height of canvas in pixels
        'camera_dist': 50.,  # distance from camera to front atom
        'image_plane': None,  # distance from front atom to image plane
        'camera_type': 'orthographic',  # perspective, ultra_wide_angle
        'point_lights': [],  # [[loc1, color1], [loc2, color2],...]
        'area_light': [(2., 3., 40.),  # location
                       'White',  # color
                       .7, .7, 3, 3],  # width, height, Nlamps_x, Nlamps_y
        'background': 'White',  # color
        'textures': None,  # length of atoms list of texture names
        'transmittances': None,  # transmittance of the atoms
        # use with care - in particular adjust the camera_distance to be closer
        'depth_cueing': False,  # fog a.k.a. depth cueing
        'cue_density': 5e-3,  # fog a.k.a. depth cueing
        'celllinewidth': 0.05,  # radius of the cylinders representing the cell
        'bondlinewidth': 0.10,  # radius of the cylinders representing bonds
        'bondatoms': [],  # [[atom1, atom2], ... ] pairs of bonding atoms
                          # For bond order > 1: [[atom1, atom2, offset,
                          #                       bond_order, bond_offset],
                          #                      ... ]
                          # bond_order: 1, 2, 3 for single, double,
                          #             and triple bond
                          # bond_offset: vector for shifting bonds from
                          #              original position. Coordinates are
                          #              in Angstrom unit.
        'exportconstraints': False}  # honour FixAtoms and mark relevant atoms?

    def __init__(self, atoms, scale=1.0, **parameters):
        for k, v in self.default_settings.items():
            setattr(self, k, parameters.pop(k, v))
        PlottingVariables.__init__(self, atoms, scale=scale, **parameters)
        constr = atoms.constraints
        self.constrainatoms = []
        for c in constr:
            if isinstance(c, FixAtoms):
                for n, i in enumerate(c.index):
                    self.constrainatoms += [i]

        self.material_styles_dict = dict(
            simple='finish {phong 0.7}',
            pale=('finish {ambient 0.5 diffuse 0.85 roughness 0.001 '
                  'specular 0.200 }'),
            intermediate=('finish {ambient 0.3 diffuse 0.6 specular 0.1 '
                          'roughness 0.04}'),
            vmd=('finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 '
                 'specular 0.5 }'),
            jmol=('finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 '
                  'metallic}'),
            ase2=('finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic '
                  'specular 0.7 roughness 0.04 reflection 0.15}'),
            ase3=('finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic '
                  'specular 1.0 roughness 0.001 reflection 0.0}'),
            glass=('finish {ambient 0.05 diffuse 0.3 specular 1.0 '
                   'roughness 0.001}'),
            glass2=('finish {ambient 0.01 diffuse 0.3 specular 1.0 '
                    'reflection 0.25 roughness 0.001}'),
        )

    def cell_to_lines(self, cell):
        return np.empty((0, 3)), None, None

    def write_ini(self, filename, **settings):
        # Determine canvas width and height
        ratio = float(self.w) / self.h
        if self.canvas_width is None:
            if self.canvas_height is None:
                self.canvas_width = min(self.w * 15, 640)
            else:
                self.canvas_width = self.canvas_height * ratio
        elif self.canvas_height is not None:
            raise RuntimeError("Can't set *both* width and height!")
        # write ini file

        ini_path = Path(filename).with_suffix('.ini')
        ini_str = f"""Input_File_Name={ini_path.with_suffix('.pov').name}
Output_to_File=True
Output_File_Type=N
Output_Alpha={'on' if self.transparent else 'off'}
; if you adjust Height, and width, you must preserve the ratio
; Width / Height = {ratio:f}
Width={self.canvas_width}
Height={self.canvas_width/ratio}
Antialias=True
Antialias_Threshold=0.1
Display={self.display}
Pause_When_Done={self.pause}
Verbose=False
"""
        with open(ini_path, 'w') as _:
            _.write(ini_str)
        return None

    def write_pov(self, filename, **settings):

        # Distance to image plane from camera
        if self.image_plane is None:
            if self.camera_type == 'orthographic':
                self.image_plane = 1 - self.camera_dist
            else:
                self.image_plane = 0
        self.image_plane += self.camera_dist


        # Produce the .pov file

        point_lights = '\n'.join(f"light_source {{{pa(loc)} {pc(rgb)}}}" 
            for loc,rgb in self.point_lights)

        area_light = ''
        if self.area_light is not None:
            loc, color, width, height, nx, ny = self.area_light
            area_light += f"""\nlight_source {{{pa(loc)} {pc(color)}
  area_light <{width:.2f}, 0, 0>, <0, {height:.2f}, 0>, {nx:n}, {ny:n}
  adaptive 1 jitter}}"""

        fog = ''
        if self.depth_cueing and (self.cue_density >= 1e-4):
            # same way vmd does it
            if self.cue_density > 1e4:
                # larger does not make any sense
                dist = 1e-4
            else:
                dist = 1. / self.cue_density
            fog += 'fog {{fog_type 1 distance {dist:.4f} color {pc(self.background)}}}'

        material_styles_dict_keys = '\n'.join(f'#declare {key} = {value}' #semicolon?
            for key, value in self.material_styles_dict.items())

        z0 = self.positions[:, 2].max() 
        self.positions -= (self.w / 2, self.h / 2, z0)

        # Draw unit cell
        if self.cell_vertices is not None:
            cell_vertices = ''
            self.cell_vertices -= (self.w / 2, self.h / 2, z0)
            self.cell_vertices.shape = (2, 2, 2, 3)
            for c in range(3):
                for j in ([0, 0], [1, 0], [1, 1], [0, 1]):
                    parts = []
                    for i in range(2):
                        j.insert(c, i)
                        parts.append(self.cell_vertices[tuple(j)])
                        del j[c]

                    distance = np.linalg.norm(parts[1] - parts[0])
                    if distance < 1e-12:
                        continue

                    cell_vertices += f'cylinder {{{pa(parts[0])}, {pa(parts[1])}, '\
                                     f'Rcell pigment {{Black}}}}\n' # all strings are f-strings for consistencey
            cell_vertices = cell_vertices.strip('\n')

        # Draw atoms
        a = 0
        atoms = ''
        for loc, dia, color in zip(self.positions, self.d, self.colors):
            tex = 'ase3'
            trans = 0.
            if self.textures is not None:
                tex = self.textures[a]
            if self.transmittances is not None:
                trans = self.transmittances[a]
            atoms += f'atom({pa(loc)}, {dia/2.:.2f}, {pc(color)}, '\
                     f'{trans}, {tex}) // #{a:n}\n'
            a += 1
        atoms = atoms.strip('\n')

        # Draw atom bonds
        bondatoms = ''
        for pair in self.bondatoms:
            # Make sure that each pair has 4 componets: a, b, offset,
            #                                           bond_order, bond_offset
            # a, b: atom index to draw bond
            # offset: original meaning to make offset for mid-point.
            # bond_oder: if not supplied, set it to 1 (single bond).
            #            It can be  1, 2, 3, corresponding to single,
            #            double, triple bond
            # bond_offset: displacement from original bond position.
            #              Default is (bondlinewidth, bondlinewidth, 0)
            #              for bond_order > 1.
            if len(pair) == 2:
                a, b = pair
                offset = (0, 0, 0)
                bond_order = 1
                bond_offset = (0, 0, 0)
            elif len(pair) == 3:
                a, b, offset = pair
                bond_order = 1
                bond_offset = (0, 0, 0)
            elif len(pair) == 4:
                a, b, offset, bond_order = pair
                bond_offset = (self.bondlinewidth, self.bondlinewidth, 0)
            elif len(pair) > 4:
                a, b, offset, bond_order, bond_offset = pair
            else:
                raise RuntimeError('Each list in bondatom must have at least '
                                   '2 entries. Error at %s' % pair)

            if len(offset) != 3:
                raise ValueError('offset must have 3 elements. '
                                 'Error at %s' % pair)
            if len(bond_offset) != 3:
                raise ValueError('bond_offset must have 3 elements. '
                                 'Error at %s' % pair)
            if bond_order not in [0, 1, 2, 3]:
                raise ValueError('bond_order must be either 0, 1, 2, or 3. '
                                 'Error at %s' % pair)

            # Up to here, we should have all a, b, offset, bond_order,
            # bond_offset for all bonds.

            # Rotate bond_offset so that its direction is 90 degree off the bond
            # Utilize Atoms object to rotate
            if bond_order > 1 and np.linalg.norm(bond_offset) > 1.e-9:
                tmp_atoms = Atoms('H3')
                tmp_atoms.set_cell(self.cell)
                tmp_atoms.set_positions([
                    self.positions[a],
                    self.positions[b],
                    self.positions[b] + np.array(bond_offset),
                ])
                tmp_atoms.center()
                tmp_atoms.set_angle(0, 1, 2, 90)
                bond_offset = tmp_atoms[2].position - tmp_atoms[1].position

            R = np.dot(offset, self.cell)
            mida = 0.5 * (self.positions[a] + self.positions[b] + R)
            midb = 0.5 * (self.positions[a] + self.positions[b] - R)
            if self.textures is not None:
                texa = self.textures[a]
                texb = self.textures[b]
            else:
                texa = texb = 'ase3'

            if self.transmittances is not None:
                transa = self.transmittances[a]
                transb = self.transmittances[b]
            else:
                transa = transb = 0.

            # draw bond, according to its bond_order.
            # bond_order == 0: No bond is plotted
            # bond_order == 1: use original code
            # bond_order == 2: draw two bonds, one is shifted by bond_offset/2,
            #                  and another is shifted by -bond_offset/2.
            # bond_order == 3: draw two bonds, one is shifted by bond_offset,
            #                  and one is shifted by -bond_offset, and the
            #                  other has no shift.
            # To shift the bond, add the shift to the first two coordinate in
            # write statement.

            posa = self.positions[a]; posb = self.positions[b]
            cola = self.colors[a]; colb = self.colors[b]

            if bond_order == 1:
                draw_tuples = (posa, mida, cola, transa, texa),\
                              (posb, midb, colb, transb, texb)  

            elif bond_order == 2:
                bs = [x / 2 for x in bond_offset]
                draw_tuples = (posa-bs, mida-bs, cola, transa, texa),\
                              (posb-bs, midb-bs, colb, transb, texb),\
                              (posa+bs, mida+bs, cola, transa, texa),\
                              (posb+bs, midb+bs, colb, transb, texb)

            elif bond_order == 3:
                bs = bond_offset
                draw_tuples = (posa   , mida   , cola, transa, texa),\
                              (posb   , midb   , colb, transb, texb),\
                              (posa+bs, mida+bs, cola, transa, texa),\
                              (posb+bs, midb+bs, colb, transb, texb),\
                              (posa-bs, mida-bs, cola, transa, texa),\
                              (posb-bs, midb-bs, colb, transb, texb)

            bondatoms += ''.join(f'cylinder {{{pa(p)}, '
                       f'{pa(m)}, Rbond texture{{pigment '
                       f'{{color {pc(c)} '
                       f'transmit {tr}}} finish{{{tx}}}}}}}\n'
                       for p, m, c, tr, tx in
                       draw_tuples)

        bondatoms = bondatoms.strip('\n') 

        # Draw constraints if requested
        constraints = ''
        if self.exportconstraints:
            for a in self.constrainatoms:
                dia = self.d[a]
                loc = self.positions[a]
                trans = 0.0
                if self.transmittances is not None:
                    trans = self.transmittances[a]
                constraints += f'constrain({pa(loc)}, {dia/2.:.2f}, Black, '\
                f'{trans}, {tex}) // #{a:n} \n' 
        constraints = constraints.strip('\n')

        pov = f"""#include "colors.inc"
#include "finish.inc"

global_settings {{assumed_gamma 1 max_trace_level 6}}
background {{{pc(self.background)}{' transmit 1.0' if self.transparent else ''}}}
camera {{{self.camera_type}
  right {self.w:.2f}*x up {self.h:.2f}*y   
  direction {self.image_plane:.2f}*z
  location <0,0,{self.camera_dist:.2f}> look_at <0,0,0>}}
{point_lights}
{area_light if area_light is not '' else '// no area light'}
{fog if fog is not '' else '// no fog'}
{material_styles_dict_keys}
#declare Rcell = {self.celllinewidth:.3f};
#declare Rbond = {self.bondlinewidth:.3f};

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{{LOC, R texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{{torus{{R, Rcell rotate 45*z texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}
     torus{{R, Rcell rotate -45*z texture{{pigment{{color COL transmit TRANS}} finish{{FIN}}}}}}
     translate LOC}}
#end

{cell_vertices if cell_vertices is not '' else '// no cell vertices'} 
{atoms}
{bondatoms}
{constraints if constraints is not '' else '// no constraints'}
""" 

#from docs: The POV-Ray language consists of identifiers, reserved keywords, floating point expressions, strings, special symbols and comments. The text of a POV-Ray scene file is free format. You may put statements on separate lines or on the same line as you desire. You may add blank lines, spaces or indentations as long as you do not split any keywords or identifiers.

        pov_path = Path(filename).with_suffix('.pov')
        pov_fid =  open(pov_path, 'w')
        pov_fid.write(pov)
        #if self.isosurface_data is not None:
        #    pov += self.add_isosurface_to_pov()
        return pov_fid


def add_isosurface_to_pov(pov_fid, pov_obj,
                          density_grid, cut_off,
                          closed_edges=False, gradient_ascending=False,
                          color=(0.85, 0.80, 0.25, 0.2), material='ase3',
                          verbose=False):
    """Computes an isosurface from a density grid and adds it to a .pov file.

    Parameters:

    pov_fid: file identifer
        The file identifer of the .pov file to be written to
    pov_obj: POVRAY instance
        The POVRAY instance that is used for writing the atoms etc.
    density_grid: 3D float ndarray
        A regular grid on that spans the cell. The first dimension corresponds
        to the first cell vector and so on.
    cut_off: float
        The density value of the isosurface.
    closed_edges: bool
        Setting this will fill in isosurface edges at the cell boundaries.
        Filling in the edges can help with visualizing highly porous structures.
    gradient_ascending: bool
        Lets you pick the area you want to enclose, i.e., should the denser
        or less dense area be filled in.
    color: povray color string, float, or float tuple
        1 float is interpreted as grey scale, a 3 float tuple is rgb, 4 float
        tuple is rgbt, and 5 float tuple is rgbft, where t is transmission
        fraction and f is filter fraction. Named Povray colors are set in
        colors.inc (http://wiki.povray.org/content/Reference:Colors.inc)
    material: string
        Can be a finish macro defined by POVRAY.material_styles or a full Povray
        material {...} specification. Using a full material specification will
        override the color patameter.

    Example:
    material = '''
      material { // This material looks like pink jelly
        texture {
          pigment { rgbt <0.8, 0.25, 0.25, 0.5> }
          finish{ diffuse 0.85 ambient 0.99 brilliance 3 specular 0.5 roughness 0.001
            reflection { 0.05, 0.98 fresnel on exponent 1.5 }
            conserve_energy
          }
        }
        interior { ior 1.3 }
      }
      photons {
          target
          refraction on
          reflection on
          collect on
      }'''
    """  # noqa: E501

    rho = density_grid
    cell = pov_obj.cell
    POV_cell_origin = pov_obj.cell_vertices[0, 0, 0]

    # print(POV_cell_disp)
    from skimage import measure
    import numpy as np
    # Use marching cubes to obtain the surface mesh of this density grid
    if gradient_ascending:
        gradient_direction = 'ascent'
        cv = 2 * cut_off
    else:
        gradient_direction = 'descent'
        cv = 0

    if closed_edges:
        shape_old = rho.shape
        # since well be padding, we need to keep the data at origin
        POV_cell_origin += -(1.0 / np.array(shape_old)) @ cell

        rho = np.pad(rho, pad_width=(1,), mode='constant', constant_values=cv)
        shape_new = rho.shape
        s = np.array(shape_new) / np.array(shape_old)
        cell = cell @ np.diag(s)

    spacing = tuple(1.0 / np.array(rho.shape))
    scaled_verts, faces, normals, values = measure.marching_cubes_lewiner(
        rho,
        level=cut_off,
        spacing=spacing,
        gradient_direction=gradient_direction,
        allow_degenerate=False,
    )

    # The verts are scaled by default, this is the super easy way of
    # distributing them in real space but it's easier to do affine
    # transformations/rotations on a unit cube so I leave it like that
    # verts = scaled_verts.dot(atoms.get_cell())
    verts = scaled_verts

    # some prime numbers for debugging formatting of lines
    # verts = verts[:31]
    # faces = faces[:47]

    if verbose:
        print('faces', len(faces))
        print('verts', len(verts))

    def wrapped_triples_section(name, triple_list,
                                triple_format="<%f, %f, %f>, ",
                                triples_per_line=4):

        pov_fid.write('\n  %s {  %i,' % (name, len(triple_list)))

        last_line_index = len(triple_list) // triples_per_line - 1
        if (len(triple_list) % triples_per_line) > 0:
            last_line_index += 1

        # print('vertex lines', last_line_index)
        for line_index in range(last_line_index + 1):
            pov_fid.write('\n      ')
            line = ''
            index_start = line_index * triples_per_line
            index_end = (line_index + 1) * triples_per_line
            # cut short if its at the last line
            index_end = min(index_end, len(triple_list))

            for index in range(index_start, index_end):
                line = line + triple_format % tuple(triple_list[index])

            if last_line_index == line_index:
                line = line[:-2] + '\n  }'

            pov_fid.write(line)

    # Start writing the mesh2
    pov_fid.write('\n\nmesh2 {')

    # the vertex_vectors (floats) and the face_indices (ints)
    wrapped_triples_section(name="vertex_vectors", triple_list=verts,
                            triple_format="<%f, %f, %f>, ", triples_per_line=4)

    wrapped_triples_section(name="face_indices", triple_list=faces,
                            triple_format="<%i, %i, %i>, ", triples_per_line=5)

    # pigment and material

    if material in pov_obj.material_styles_dict.keys():
        material = '''
  material {
    texture {
      pigment { %s }
      finish { %s }
    }
  }''' % (pc(color), material)
    pov_fid.writelines(material)

    # now for the rotations of the cell
    matrix_transform = [
        '\n  matrix < %f, %f, %f,' % tuple(cell[0]),
        '\n        %f, %f, %f,' % tuple(cell[1]),
        '\n        %f, %f, %f,' % tuple(cell[2]),
        '\n        %f, %f, %f>' % tuple(POV_cell_origin),
    ]
    pov_fid.writelines(matrix_transform)

    # close the brackets
    pov_fid.writelines('\n}\n')

    # pov_fid.close()


def write_povray_input(filename, atoms, extras=[], **parameters):
    if isinstance(atoms, list):
        assert len(atoms) == 1
        atoms = atoms[0]
    assert 'scale' not in parameters
    pov_obj = POVRAY(atoms, **parameters)
    pov_obj.write_ini(filename) # returns none
    pov_obj.write_pov(filename) # returns open file
    # evaluate and write extras
    #for function, params in extras:
    #    function(pov_fid, pov_obj, **params)
    # the povray file wasn't explicitly being closed before the addition
    # of the extras option.
    #pov_fid.close()

def run_pov(filename, povray_executable='povray',stderr=None):
    ini_path = Path(filename).with_suffix('.ini')
    cmd = [povray_executable, ini_path.as_posix()]
    if stderr != '-':
        if stderr is None:
            check_call(cmd, stderr=DEVNULL)
        else:
            with open(stderr, 'w') as stderr:
                check_call(cmd, stderr=stderr)
    else:
        check_call(cmd)

if __name__ == '__main__':
    from ase.build import molecule
    H2 = molecule('H2')
    write_povray_input('H2.pov', H2)
