import warnings
from math import pi, sin, cos
import numpy as np


def bz_vertices(icell, dim=3):
    """See https://xkcd.com/1421 ..."""
    from scipy.spatial import Voronoi
    icell = icell.copy()
    if dim < 3:
        icell[2, 2] = 1e-3
    if dim < 2:
        icell[1, 1] = 1e-3

    indices = (np.indices((3, 3, 3)) - 1).reshape((3, 27))
    G = np.dot(icell.T, indices).T
    vor = Voronoi(G)
    bz1 = []
    for vertices, points in zip(vor.ridge_vertices, vor.ridge_points):
        if -1 not in vertices and 13 in points:
            normal = G[points].sum(0)
            normal /= (normal**2).sum()**0.5
            bz1.append((vor.vertices[vertices], normal))
    return bz1


class BZFlatPlot:
    axis_dim = 2  # Dimension of the plotting surface (2 even if it's 1D BZ).

    def new_axes(self, fig):
        return fig.gca()

    def adjust_view(self, ax, minp, maxp):
        ax.autoscale_view(tight=True)
        s = maxp * 1.05
        ax.set_xlim(-s, s)
        ax.set_ylim(-s, s)
        ax.set_aspect('equal')

    def draw_arrow(self, ax, vector, **kwargs):
        ax.arrow(0, 0, vector[0], vector[1],
                 lw=1,
                 length_includes_head=True,
                 head_width=0.03,
                 head_length=0.05,
                 **kwargs)

    def label_options(self, point):
        ha_s = ['right', 'left', 'right']
        va_s = ['bottom', 'bottom', 'top']

        x, y = point
        ha = ha_s[int(np.sign(x))]
        va = va_s[int(np.sign(y))]
        return {'ha': ha, 'va': va, 'zorder': 4}

    point_options = {'zorder': 5}


class BZSpacePlot:
    axis_dim = 3
    point_options = {}

    def __init__(self, *, elev=None):
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d import proj3d
        from matplotlib.patches import FancyArrowPatch
        Axes3D  # silence pyflakes

        class Arrow3D(FancyArrowPatch):
            def __init__(self, ax, xs, ys, zs, *args, **kwargs):
                FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
                self._verts3d = xs, ys, zs
                self.ax = ax

            def draw(self, renderer):
                xs3d, ys3d, zs3d = self._verts3d
                xs, ys, zs = proj3d.proj_transform(xs3d, ys3d,
                                                   zs3d, self.ax.axes.M)
                self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
                FancyArrowPatch.draw(self, renderer)

            # FIXME: Compatibility fix for matplotlib 3.5.0: Handling of 3D
            # artists have changed and all 3D artists now need
            # "do_3d_projection". Since this class is a hack that manually
            # projects onto the 3D axes we don't need to do anything in this
            # method. Ideally we shouldn't resort to a hack like this.
            def do_3d_projection(self, *_, **__):
                return 0

        self.arrow3d = Arrow3D
        self.azim = pi / 5
        if elev is None:
            elev = pi / 6
        self.elev = elev
        x = sin(self.azim)
        y = cos(self.azim)
        self.view = [x * cos(elev), y * cos(elev), sin(elev)]

    def new_axes(self, fig):
        return fig.add_subplot(projection='3d')

    def draw_arrow(self, ax, vector, **kwargs):
        ax.add_artist(self.arrow3d(
            ax,
            [0, vector[0]],
            [0, vector[1]],
            [0, vector[2]],
            mutation_scale=20,
            arrowstyle='-|>',
            **kwargs))

    def adjust_view(self, ax, minp, maxp):
        import matplotlib.pyplot as plt
        # ax.set_aspect('equal') <-- won't work anymore in 3.1.0
        ax.view_init(azim=self.azim / pi * 180, elev=self.elev / pi * 180)
        # We want aspect 'equal', but apparently there was a bug in
        # matplotlib causing wrong behaviour.  Matplotlib raises
        # NotImplementedError as of v3.1.0.  This is a bit unfortunate
        # because the workarounds known to StackOverflow and elsewhere
        # all involve using set_aspect('equal') and then doing
        # something more.
        #
        # We try to get square axes here by setting a square figure,
        # but this is probably rather inexact.
        fig = ax.get_figure()
        xx = plt.figaspect(1.0)
        fig.set_figheight(xx[1])
        fig.set_figwidth(xx[0])

        # oldlibs tests with matplotlib 2.0.0 say we have no set_proj_type:
        if hasattr(ax, 'set_proj_type'):
            ax.set_proj_type('ortho')

        minp0 = 0.9 * minp  # Here we cheat a bit to trim spacings
        maxp0 = 0.9 * maxp
        ax.set_xlim3d(minp0, maxp0)
        ax.set_ylim3d(minp0, maxp0)
        ax.set_zlim3d(minp0, maxp0)

        if hasattr(ax, 'set_box_aspect'):
            ax.set_box_aspect([1, 1, 1])
        else:
            msg = ('Matplotlib axes have no set_box_aspect() method.  '
                   'Aspect ratio will likely be wrong.  '
                   'Consider updating to Matplotlib >= 3.3.')
            warnings.warn(msg)

    def label_options(self, point):
        return dict(ha='center', va='bottom')


def normalize_name(name):
    if name == 'G':
        return '\\Gamma'

    if len(name) > 1:
        import re
        m = re.match(r'^(\D+?)(\d*)$', name)
        if m is None:
            raise ValueError('Bad label: {}'.format(name))
        name, num = m.group(1, 2)
        if num:
            name = f'{name}_{{{num}}}'
    return name


def bz_plot(cell, vectors=False, paths=None, points=None,
            elev=None, scale=1, interactive=False,
            pointstyle=None, ax=None, show=False):
    import matplotlib.pyplot as plt

    if pointstyle is None:
        pointstyle = {}

    cell = cell.copy()

    dimensions = cell.rank
    if dimensions == 3:
        plotter = BZSpacePlot()
    else:
        plotter = BZFlatPlot()
    assert dimensions > 0, 'No BZ for 0D!'

    if ax is None:
        ax = plotter.new_axes(plt.gcf())

    assert not cell[dimensions:, :].any()
    assert not cell[:, dimensions:].any()

    icell = cell.reciprocal()
    kpoints = points
    bz1 = bz_vertices(icell, dim=dimensions)

    maxp = 0.0
    minp = 0.0
    if dimensions == 1:
        length = icell[0, 0]
        x = 0.5 * length * np.array([-1, 0, -1])
        ax.plot(x, np.zeros(3), c='k', ls='-')
        maxp = length
    else:
        for points, normal in bz1:
            ls = '-'
            xyz = np.concatenate([points, points[:1]]).T
            if dimensions == 3:
                if normal @ plotter.view < 0 and not interactive:
                    ls = ':'

            ax.plot(*xyz[:plotter.axis_dim], c='k', ls=ls)
            maxp = max(maxp, points.max())
            minp = min(minp, points.min())

    if vectors:
        for i in range(dimensions):
            plotter.draw_arrow(ax, icell[i], color='k')

        # XXX Can this be removed?
        if dimensions == 3:
            maxp = max(maxp, 0.6 * icell.max())
        else:
            maxp = max(maxp, icell.max())

    if paths is not None:
        for names, points in paths:
            coords = np.array(points).T[:plotter.axis_dim, :]
            ax.plot(*coords, c='r', ls='-')

            for name, point in zip(names, points):
                name = normalize_name(name)
                point = point[:plotter.axis_dim]
                ax.text(*point, rf'$\mathrm{{{name}}}$',
                        color='g', **plotter.label_options(point))

    if kpoints is not None:
        kw = {'c': 'b', **plotter.point_options, **pointstyle}
        ax.scatter(*kpoints[:, :plotter.axis_dim].T, **kw)

    ax.set_axis_off()

    plotter.adjust_view(ax, minp, maxp)

    if show:
        plt.show()

    return ax
