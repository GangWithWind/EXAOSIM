import numpy as np
from math import factorial
from scipy.interpolate import RectBivariateSpline as rbs

#note!! the zernike function is from https://poppy-optics.readthedocs.io/en/stable/_modules/poppy/zernike.html
#without lissions, so don't open source it


def _is_odd(integer):
    """Helper for testing if an integer is odd by bitwise & with 1."""
    return integer & 1

def noll_indices(j):
    """Convert from 1-D to 2-D indexing for Zernikes or Hexikes.

    Parameters
    
    j : int
        Zernike function ordinate, following the convention of Noll et al. JOSA 1976.
        Starts at 1.

    """

    if j < 1:
        raise ValueError("Zernike index j must be a positive integer.")

    # from i, compute m and n
    # I'm not sure if there is an easier/cleaner algorithm or not.
    # This seems semi-complicated to me...

    # figure out which row of the triangle we're in (easy):
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        nprev = (n + 1) * (n + 2) / 2  # figure out which entry in the row (harder)
        # The rule is that the even Z obtain even indices j, the odd Z odd indices j.
        # Within a given n, lower values of m obtain lower j.

        resid = int(j - nprev - 1)

        if _is_odd(j):
            sign = -1
        else:
            sign = 1

        if _is_odd(n):
            row_m = [1, 1]
        else:
            row_m = [0]

        for i in range(int(np.floor(n / 2.))):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[resid] * sign

    # _log.debug("J=%d:\t(n=%d, m=%d)" % (j, n, m))
    return n, m

def R(n, m, rho):
    """Compute R[n, m], the Zernike radial polynomial

    Parameters
    ----------
    n, m : int
        Zernike function degree
    rho : array
        Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the
        unit circle
    """

    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
            coef = ((-1) ** k * factorial(n - k) /
                    (factorial(k) * factorial((n + m) / 2. - k) * factorial((n - m) / 2. - k)))
            output += coef * rho ** (n - 2 * k)
        return output


def zernike(n, m, npix=100, rho=None, theta=None, outside=np.nan,
            noll_normalize=True, **kwargs):
    """Return the Zernike polynomial Z[m,n] for a given pupil.

    For this function the desired Zernike is specified by 2 indices m and n.
    See zernike1 for an equivalent function in which the polynomials are
    ordered by a single index.

    You may specify the pupil in one of two ways:

        zernike(n, m, npix)
            where npix specifies a pupil diameter in pixels.
            The returned pupil will be a circular aperture
            with this diameter, embedded in a square array
            of size npix*npix.

        zernike(n, m, rho=r, theta=theta)
            Which explicitly provides the desired pupil coordinates
            as arrays r and theta. These need not be regular or contiguous.

    The expressions for the Zernike terms follow the normalization convention
    of Noll et al. JOSA 1976 unless the `noll_normalize` argument is False.

    Parameters
    ----------
    n, m : int
        Zernike function degree
    npix : int
        Desired diameter for circular pupil. Only used if `rho` and
        `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.
    outside : float
        Value for pixels outside the circular aperture (rho > 1).
        Default is `np.nan`, but you may also find it useful for this to
        be 0.0 sometimes.
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernike definition is
        modified such that the integral of Z[n, m] * Z[n, m] over the
        unit disk is pi exactly. To omit the normalization constant,
        set this to False. Default is True.

    Returns
    -------
    zern : 2D numpy array
        Z(m,n) evaluated at each (rho, theta)
    """
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        print("Radial polynomial is zero for these inputs: m={}, n={} "
                  "(are you sure you wanted this Zernike?)".format(m, n))
    # print("Zernike(n=%d, m=%d)" % (n, m))

    if theta is None and rho is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                         "provide both of them.")

    if not np.all(rho.shape == theta.shape):
        raise ValueError('The rho and theta arrays do not have consistent shape.')

    aperture = np.ones(rho.shape)
    aperture[np.where(rho > 1)] = 0.0  # this is the aperture mask

    if m == 0:
        if n == 0:
            zernike_result = aperture
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture

    zernike_result[np.where(rho > 1)] = outside
    return zernike_result


def zernike_xy(n, m, xx, yy, outside=np.nan,
            noll_normalize=True, **kwargs):
    """Return the Zernike polynomial Z[m,n] for a given pupil.

    For this function the desired Zernike is specified by 2 indices m and n.
    See zernike1 for an equivalent function in which the polynomials are
    ordered by a single index.

    You may specify the pupil in one of two ways:

        zernike(n, m, npix)
            where npix specifies a pupil diameter in pixels.
            The returned pupil will be a circular aperture
            with this diameter, embedded in a square array
            of size npix*npix.

        zernike(n, m, rho=r, theta=theta)
            Which explicitly provides the desired pupil coordinates
            as arrays r and theta. These need not be regular or contiguous.

    The expressions for the Zernike terms follow the normalization convention
    of Noll et al. JOSA 1976 unless the `noll_normalize` argument is False.

    Parameters
    ----------
    n, m : int
        Zernike function degree
    npix : int
        Desired diameter for circular pupil. Only used if `rho` and
        `theta` are not provided.
    rho, theta : array_like
        Image plane coordinates. `rho` should be 0 at the origin
        and 1.0 at the edge of the circular pupil. `theta` should be
        the angle in radians.
    outside : float
        Value for pixels outside the circular aperture (rho > 1).
        Default is `np.nan`, but you may also find it useful for this to
        be 0.0 sometimes.
    noll_normalize : bool
        As defined in Noll et al. JOSA 1976, the Zernike definition is
        modified such that the integral of Z[n, m] * Z[n, m] over the
        unit disk is pi exactly. To omit the normalization constant,
        set this to False. Default is True.

    Returns
    -------
    zern : 2D numpy array
        Z(m,n) evaluated at each (rho, theta)
    """
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        print("Radial polynomial is zero for these inputs: m={}, n={} "
                  "(are you sure you wanted this Zernike?)".format(m, n))
    # print("Zernike(n=%d, m=%d)" % (n, m))



    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    aperture = np.ones(rho.shape)
    aperture[np.where(rho > 1)] = 0.0  # this is the aperture mask

    if m == 0:
        if n == 0:
            zernike_result = aperture
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture

    zernike_result[np.where(rho > 1)] = outside
    return zernike_result

def PR(n, m, rho):
    """Compute patial R[n, m]/patial x, and patial R[n, m]/patial y

    Parameters
    ----------
    n, m : int
        Zernike function degree
    rho : array
        Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the
        unit circle
    """

    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
            coef = ((-1) ** k * factorial(n - k) /
                    (factorial(k) * factorial((n + m) / 2. - k) * factorial((n - m) / 2. - k)))
            if ((n - 2 * k) != 0):
                output += coef * rho ** (n - 2 * k - 1) * (n - 2 * k)
        return output

    
def ROR(n, m, rho):
    """Compute R[n, m]/rho

    Parameters
    ----------
    n, m : int
        Zernike function degree
    rho : array
        Image plane radial coordinates. `rho` should be 1 at the desired pixel radius of the
        unit circle
    """

    m = int(np.abs(m))
    n = int(np.abs(n))
    output = np.zeros(rho.shape)
    if _is_odd(n - m):
        return 0
    else:
        for k in range(int((n - m) / 2) + 1):
            coef = ((-1) ** k * factorial(n - k) /
                    (factorial(k) * factorial((n + m) / 2. - k) * factorial((n - m) / 2. - k)))
            output += coef * rho ** (n - 2 * k - 1)
        return output


def Pzernike_xy(n, m, x, y,outside=0,noll_normalize=True):
    
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        print("Radial polynomial is zero for these inputs: m={}, n={} "
                  "(are you sure you wanted this Zernike?)".format(m, n))
    # print("Zernike(n=%d, m=%d)" % (n, m))

    xx = np.array(x)
    yy = np.array(y)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)

    if m == 0:
        if n == 0:
            zc = 0
            zc2 = 0
        else:
            zc = PR(n, m, rho) / np.sqrt(2)
            zc2 = 0
            
    elif m > 0:
        zc = PR(n, m, rho) * np.cos(np.abs(m) * theta)
        zc2 = - ROR(n, m, rho) * np.sin(np.abs(m) * theta) * np.abs(m)
        
    else:
        zc = PR(n, m, rho) * np.sin(np.abs(m) * theta)
        zc2 = ROR(n, m, rho) * np.cos(np.abs(m) * theta) * np.abs(m)

    zcx = zc * np.cos(theta) - zc2 * np.sin(theta)
    zcy = zc * np.sin(theta) + zc2 * np.cos(theta)

    norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
    zrx = norm_coeff * zcx
    zrx[np.where(rho > 1)] = outside
    
    zry = norm_coeff * zcy
    zry[np.where(rho > 1)] = outside

    return zrx, zry


def Pzernike(n, m, npix=100, rho=None, theta=None, outside=np.nan,
            noll_normalize=True, **kwargs):
    
    if not n >= m:
        raise ValueError("Zernike index m must be >= index n")
    if (n - m) % 2 != 0:
        print("Radial polynomial is zero for these inputs: m={}, n={} "
                  "(are you sure you wanted this Zernike?)".format(m, n))
    # print("Zernike(n=%d, m=%d)" % (n, m))

    if theta is None and rho is None:
        x = (np.arange(npix, dtype=np.float64) - (npix - 1) / 2.) / ((npix - 1) / 2.)
        y = x
        xx, yy = np.meshgrid(x, y)

        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(yy, xx)
    elif (theta is None and rho is not None) or (theta is not None and rho is None):
        raise ValueError("If you provide either the `theta` or `rho` input array, you must "
                         "provide both of them.")

    if not np.all(rho.shape == theta.shape):
        raise ValueError('The rho and theta arrays do not have consistent shape.')

    aperture = np.ones(rho.shape)
    aperture[np.where(rho > 1)] = 0.0  # this is the aperture mask

    if m == 0:
        if n == 0:
            zc = 0
            zc2 = 0
        else:
            zc = PR(n, m, rho) / np.sqrt(2)
            zc2 = 0
            
    elif m > 0:
        zc = PR(n, m, rho) * np.cos(np.abs(m) * theta)
        zc2 = - ROR(n, m, rho) * np.sin(np.abs(m) * theta) * np.abs(m)
        
    else:
        zc = PR(n, m, rho) * np.sin(np.abs(m) * theta)
        zc2 = ROR(n, m, rho) * np.cos(np.abs(m) * theta) * np.abs(m)

    zcx = zc * np.cos(theta) - zc2 * np.sin(theta)
    zcy = zc * np.sin(theta) + zc2 * np.cos(theta)

    norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
    zrx = norm_coeff * zcx * aperture
    zrx[np.where(rho > 1)] = outside
    
    zry = norm_coeff * zcy * aperture
    zry[np.where(rho > 1)] = outside
    
    return zrx, zry



class ZernikeRBS(object):
    def __init__(self, phase, pix = True):
        self.data0 = phase
        
        sz = phase.shape[0]
        x = np.arange(sz) - (sz - 1)/2
        if not pix:
            x = x / (sz - 1) * 2
        
        self.fun = rbs(x, x, phase)
        
    def get(self, x, y):
#         zp = z ** self.p
        fy = self.fun(x, y, dx = 1, dy = 0)
        fx = self.fun(x, y, dx = 0, dy = 1)
            
        return (fx[0, 0], fy[0, 0])