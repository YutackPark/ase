import warnings
def extrapolate(x, y, n=-1.5, plot=0, reg=0, txt=None):
    warnings.warn(
        'The ase.utils.extrapolate module has been deprecated.  '
        'Please use gpaw.utilities.extrapolate.extrapolate() instead.',
        DeprecationWarning
    )
