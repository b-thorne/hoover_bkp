import healpy as hp
import jax
import jax.numpy as np


class WhiteNoise(object):
    """ Simple white noise model. See eqs (1) and (2) from arxiv: 9705188.
    """
    def __init__(self, weights=None, sens=None, pol=True):
        if weights is not None:
            assert weights.ndim == 1
            self.nfreq = weights.shape[0]
            self.weights = weights

        if sens is not None:
            assert sens.ndim == 1
            self.nfreq = sens.shape[0]
            self.weights = self._sens_to_weight(sens)

        self.npol = 1
        if pol:
            self.npol = 2
        return

    def map(self, nside, seed=3232):
        """ Function to generate a realization of the noise level
        as a HEALPix map, with a given `nside`.

        Parameters
        ----------
        nside: int
            Nside parameter of the HEALPix map to be produced.
        seed: int (optional, default=3232)
            PRNG seed for the `jax` key.

        Returns
        -------
        ndarray
            Noise realization as a HEALPix map.
        """
        npix = hp.nside2npix(nside)
        pix_var = self._get_pix_var(npix)
        key = jax.random.PRNGKey(seed)
        out = jax.random.normal(key, shape=(self.nfreq, self.npol, npix), dtype=np.float32)
        out *= np.sqrt(pix_var[:, None, None])
        return out

    def spectrum(self, lmax, fwhm=None):
        return self._weight_to_spec(lmax, fwhm)

    def get_pix_var_map(self, nside):
        """ Function to return a map containing the variance
        in each pixel at each frequency. This is a product
        consumed by the `hoover.likelihood.LogProb` object,
        and is provided for convenient use with that.

        Parameters
        ----------
        nside: int
            Nside parameter of the HEALPix maps to be produced.

        Returns
        -------
        ndarray
            Array containing the variance in each pixel at each frequency.
        """
        npix = hp.nside2npix(nside)
        pix_var = self._get_pix_var(npix)
        out = np.ones((self.nfreq, self.npol, npix))
        return out * pix_var[..., None, None]

    def _get_pix_var(self, npix):
        """ Pixel variance is given by:

        ..math: \sigma_{\rm pix}^2 = \frac{w^{-1}}{4 \pi } N_{\rm pix}
        """
        return self.weights * npix / 4. / np.pi

    def _sens_to_weight(self, sens):
        return 4. * np.pi * sens ** 2 / (4. * np.pi * (180. / np.pi) ** 2 * 60. ** 2)

    def _weight_to_spec(self, lmax, fwhm=None):
        """ Noise spectrum is given by:

        ..math: N_\ell = w^{-1} B^2_{\ell}

        where the beam is given by:

        ..math: B_\ell = \exp(-\ell(\ell + 1) \theta_{\rm FWHM}^2 / (8 \log 2))

        """
        ells = np.arange(2, lmax + 1)
        pol_spec = _pol_spec(self.weights[:, None], ells, fwhm)

        nell = np.zeros((self.nfreq, 2, lmax + 1))
        nell = jax.ops.index_update(nell, jax.ops.index[:, 0, 2:], pol_spec)
        nell = jax.ops.index_update(nell, jax.ops.index[:, 0, 2:], pol_spec)
        return nell

def _pol_spec(weight, ells, fwhm=None):
    """ Function to calculate white noise power spectrum with option to
    add beam effects. 

    Parameters
    ----------
    weight: ndarray
        Array containing weights of frequency channels, shape (Nfreq)
    ells: ndarray
        Array of multipoles over which to calculate the noise power spectrum.
    fwhm: float (optional, default=None) 
        If not None, deconvolve a beam with this fwhm in radians.

    Returns
    -------
    ndarray
        Power spectrum of noise.
    """
    if fwhm is not None:
        print(fwhm)
        return weight / _gaussian_beam(fwhm, ells) ** 2
    else:
        return weight * np.ones_like(ells)

def _gaussian_beam(fwhm, ells):
    """ Function to calculate a Gaussian window function in harmonic space.

    Parameters
    ----------
    fwhm: float
        Full width at half-maximum of the Gaussian beam, in radians.
    ells: ndarray
        Array containing the multipoles over which to calculate the beam.

    Returns
    -------
    ndarray
        Array containing the beam window function.
    """
    return np.exp(- ells * (ells + 1) * fwhm ** 2 / (np.log(8) * 2))
