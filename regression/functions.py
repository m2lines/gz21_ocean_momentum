import numpy as np


#grid_cm26 = get_grid()

# This defines standard functions used for sparse regression

def bz(velocity: np.ndarray):
    """
    Return the BZ parameterization
    """
    # TODO not efficient to do this every call
    grid = grid_cm26.interp(dict(xu_ocean=velocity.xu_ocean,
                            yu_ocean=velocity.yu_ocean)) * 4
    velocity = velocity / 10
    zeta = (velocity['vsurf'].diff(dim='xu_ocean') / grid['dxu']
           - velocity['usurf'].diff(dim='yu_ocean') / grid['dyu'])
    d = (velocity['usurf'].diff(dim='yu_ocean') / grid['dyu']
        + velocity['vsurf'].diff(dim='xu_ocean') / grid['dxu'])
    d_tilda = (velocity['usurf'].diff(dim='xu_ocean') / grid['dxu']
              - velocity['vsurf'].diff(dim='yu_ocean') / grid['dyu'])
    zeta_sq = zeta**2
    s_x = ((zeta_sq - zeta * d).diff(dim='xu_ocean') / grid['dxu']
            + (zeta * d_tilda).diff(dim='yu_ocean') / grid['dyu'])
    s_y = ((zeta * d_tilda).diff(dim='xu_ocean') / grid['dxu']
          + (zeta_sq + zeta * d).diff(dim='yu_ocean') / grid['dyu'])
    k_bt = -4.87 * 1e8
    s_x, s_y = s_x * 1e7 * k_bt, s_y * 1e7 * k_bt
    return s_x, s_y


def zeta(velocity: np.ndarray):
    grid = grid_cm26.interp(dict(xu_ocean=velocity.xu_ocean,
                                 yu_ocean=velocity.yu_ocean)) * 4
    velocity = velocity / 10
    zeta = (velocity['vsurf'].diff(dim='xu_ocean') / grid['dxu']
            - velocity['usurf'].diff(dim='yu_ocean') / grid['dyu'])
    return zeta * 1e7


def d(velocity: np.ndarray):
    grid = grid_cm26.interp(dict(xu_ocean=velocity.xu_ocean,
                                 yu_ocean=velocity.yu_ocean)) * 4
    d = (velocity['usurf'].diff(dim='yu_ocean') / grid['dyu']
         + velocity['vsurf'].diff(dim='xu_ocean') / grid['dxu'])
    return d * 1e7


def d_tilda(velocity: np.ndarray):
    grid = grid_cm26.interp(dict(xu_ocean=velocity.xu_ocean,
                                 yu_ocean=velocity.yu_ocean)) * 4
    d_tilda = (velocity['usurf'].diff(dim='xu_ocean') / grid['dxu']
               - velocity['vsurf'].diff(dim='yu_ocean') / grid['dyu'])
    return d_tilda * 1e7


def laplacian(velocity: np.ndarray):
    pass