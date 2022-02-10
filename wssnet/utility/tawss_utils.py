import numpy as np
from scipy.integrate import simps

def get_wss_magnitude(wss_vector):
    if (wss_vector.shape[-1] != 3):
        return wss_vector

    return np.sum(wss_vector ** 2, axis=-1) ** 0.5

def get_tawss(wss, dt):
    x = np.arange(0, len(wss))
    x = np.asarray(x) * dt

    wss_int = simps(wss, x, axis=0)
    return wss_int / (x[-1] - x[0])

    
def get_osi(wss_vector, dt):
    eps = 1e-12
    if (wss_vector.shape[-1] != 3):
        return np.zeros((wss_vector.shape[1], wss_vector.shape[2]))
    
    x = np.arange(0, len(wss_vector))
    x = np.asarray(x) * dt

    # numerator
    wss_x = simps(wss_vector[...,0], x, axis=0)
    wss_y = simps(wss_vector[...,1], x, axis=0)
    wss_z = simps(wss_vector[...,2], x, axis=0)

    wss_mag = (wss_x ** 2 + wss_y ** 2 + wss_z ** 2) ** 0.5

    # denominator
    denom = get_wss_magnitude(wss_vector)
    denom = simps(denom, x, axis=0)

    frac = (wss_mag + eps) / (denom + eps)

    osi = 0.5 * ( 1 - frac)

    return osi


def get_osi_discrete(wss_vector):
    if (wss_vector.shape[-1] != 3):
        return np.zeros((wss_vector.shape[1], wss_vector.shape[2]))
    
    # numerator
    wss_x = np.sum(wss_vector[...,0], axis=0)
    wss_y = np.sum(wss_vector[...,1], axis=0)
    wss_z = np.sum(wss_vector[...,2], axis=0)

    wss_mag = (wss_x ** 2 + wss_y ** 2 + wss_z ** 2) ** 0.5
    
    # denominator
    denom = get_wss_magnitude(wss_vector)
    denom = np.sum(denom, axis=0)
    
    frac = wss_mag / denom
    osi = 0.5 * ( 1 - frac)
    
    return osi
