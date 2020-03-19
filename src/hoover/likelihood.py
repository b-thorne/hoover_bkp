import jax.numpy as np
from jax import jit
from .seds import FMatrix


class LogProb(object):
    r""" Class calculating the marginalize posterior of spectral parameters
    given some observed data, its covariance, and a model for the emission
    of different components.

    This class links together the functions required to calculate the
    terms in Equation (A7) of 1608.00551.
    """
    def __init__(self, data, covariance, fmatrix, priors={}) :
        self._fmatrix = fmatrix
        self._preprocess_data(data, covariance)
        self._priors = priors

    def __call__(self, pars, *args, **kwargs):
        r""" This method computes the log probability at a point in parameter
        space specified by `pars` and any additional `kwargs` that may overwrite
        members of the `pars` dictionary.

        Parameters
        ----------
        pars: dictionary
            Dictionary containing all the necessary parameters to specify
            the sky model.

        Returns
        -------
        float
            The log probability at this point in parameter space.
        """
        pars.update(kwargs)
        lnprior = self._lnprior()
        F = self._F(pars)
        N_T_inv = self._N_T_inv(pars, F=F)
        T_bar = self._T_bar(pars, F=F, N_T_inv=N_T_inv)
        return _lnP(lnprior, T_bar, N_T_inv)

    def get_amplitude_expectation(self, pars, *args, **kwargs):
        r""" Convenience function to return the component-separated expected
        amplitudes, `T_bar`, taking care of the relevant reshaping.
        """
        T_bar = self._T_bar(pars, *args, **kwargs)
        return np.moveaxis(T_bar.reshape(self.npol, self.npix, -1), 2, 0)

    def get_amplitdue_covariance(self, pars, *args, **kwargs):
        r""" Convenience function to return the component covariances,
        `N_T_inv`, for a given set of spectral parameters.
        """
        # NOT IMPLEMENTED
        #return self._N_T_inv(pars, *args, **kwargs)

    def _preprocess_data(self, data, covariance):
        r""" This function does some preprocessing of the observed data
        and covariance.

        We reshape the data from (Nfreq, Npol, Npix) to (Npol * Npix, Nfreq).

        Parameters
        ----------
        data: ndarray
            Array of shape (Nfreq, Npol, Npix) containing observed sky.

        cov: ndarray
            Array of shape (Nfreqs, Npol, Npix) containing the noise covariance
            of the observations `data`.

        Returns
        -------
        None
        """
        try:
            assert data.ndim == 3
        except AssertionError:
            raise AssertionError("Data must have three dimensions, Nfreq, Npol, Npix")
        self.nfreq, self.npol, self.npix = data.shape
        self.data_shape = [self.npix * self.npol, self.nfreq]
        self.data = self._reorder_reshape_inputs(data)
        self.N_inv = 1. / self._reorder_reshape_inputs(covariance) #Inverse variance
        self.N_inv_d = self.data * self.N_inv #Inverse variance-weighted data

    def _reorder_reshape_inputs(self, arr):
        r""" Function to reorder axes and reshape dimensions of input data.

        This takes input data, assumed to be of shape: (Nfreq, Npol, Npix)
        and converts to shape (Npix * Npol, Nfreq), which is easier to work
        with in the likelihood.

        Parameters
        ----------
        arr: ndarray
            Numpy array with three dimensions.

        Returns
        -------
        ndarray
            Numpy array with two dimensions.
        """
        return np.moveaxis(arr, (0, 1, 2), (2, 0, 1)).reshape(self.data_shape).astype(np.float32)

    def _F(self, pars, *args, **kwargs):
        pars.update(kwargs)
        return self._fmatrix(**pars)

    def _N_T_inv(self, pars, F=None, *args, **kwargs):
        pars.update(kwargs)
        if F is None:
            F = self._F(pars)
        return _N_T_inv(F, self.N_inv)

    def _T_bar(self, pars, F=None, N_T_inv=None, *args, **kwargs):
        pars.update(kwargs)
        if F is None:
            F = self._F(pars)
        if N_T_inv is None:
            N_T_inv = self._N_T_inv(pars, F=F)
        return _T_bar(F, N_T_inv, self.N_inv_d)

    def _lnprior(self, *args, **kwargs):
        logprior = 0
        for key, (mean, std) in self._priors.items():
            par = kwargs.get(key, None)
            if par is None:
                pass
            else:
                logprior += _log_gaussian(par, mean, std)
        return logprior


@jit
def _N_T_inv(F: np.ndarray, N_inv: np.ndarray) -> np.ndarray:
    r"""Function to calculate the inverse covariance of component
    amplitudes, `N_T_inv. This is an implementation of Equation
    (A4) in 1608.00551. See also Equation (A10) for interpretation.

    Parameters
    ----------
    F: ndarray
        SED matrix
    N_inv: ndarray
        Inverse noise covariance.

    Returns
    -------
    ndarray
        N_T_inv, the inverse covariance of component amplitude.
    """
    Fprod = F[:, None, :] * F[None, :, :]
    return np.sum(Fprod[None, :, :, :] * N_inv[:, None, None, :], axis=3)


@jit
def _T_bar(F: np.ndarray, N_T_inv: np.ndarray, N_inv_d: np.ndarray) -> np.ndarray:
    r"""Function to calculate the expected component amplitudes, `T_bar`.
    This is an implementation of Equation (A4) in 1608.00551. See also
    Equation (A10) for interpretation.

    Parameters
    ----------
    F: ndarray
        SED matrix
    N_T_inv: ndarray
        Inverse component covariance.
    N_inv_d: ndarray
        Inverse covariance-weighted data.

    Returns
    -------
    ndarray
        T_bar, the expected component amplitude.
    """
    y = np.sum(F[None, :, :] * N_inv_d[:, None, :], axis=2)
    return np.linalg.solve(N_T_inv, y)


@jit
def _lnP(lnprior, T_bar, N_T_inv):
    r""" Function to calculate the posterior marginalized over
    amplitude parameters.

    This function calcualtes Equation (A7), with the inclusion of a
    Jeffrey's prior of 1608.00551:

    ..math: p(b|d) \propto \exp\left[\frac{1}{2}\bar{T}^T N_T^{-1} \bar{T} \right]p_p(b)

    Parameters
    ----------
    lnprior: float
        Prior evaluated at the given point in parameter space
    T_bar: ndarray
        Array containing the component amplitude means calculated at this
        point in parameter space.
    N_T_inv: ndarray
        Component amplitude covariance calculated at this point in
        parameter space.

    Returns
    -------
    float
        Log likelihood of this set of spectral parameters.
    """
    return lnprior + 0.5 * np.einsum("ij,ijk,ik->", T_bar, N_T_inv, T_bar)


@jit
def _log_gaussian(par, mean, std):
    r""" Function used to calculate a Gaussian prior.
    """
    return 0.5 * ((par - mean) / std) ** 2
