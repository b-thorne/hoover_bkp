import yaml
import h5py

import jax.numpy as np
from jax import jit
import jax
from .seds import FMatrix

__all__ = ['LogProb']

class LogProb(object):
    r""" Class calculating the marginalize posterior of spectral parameters
    given some observed data, its covariance, and a model for the emission
    of different components.

    This class links together the functions required to calculate the
    terms in Equation (A7) of 1608.00551.
    """
    def __init__(self, data=None, covariance=None, frequencies=None, model=None):
        r""" The setup of this class requires the observed multifrequency
        sky maps, their pixel covariance (assumed diagonal in pixel space), 
        the SED matrix (instance of `hoover.FMatrix`), a dictionary
        containing the fixed parameter and their values, and a set of priors,
        which tell the likelihood which parameters are to be let vary.

        Parameters
        ----------
        data: ndarray
            Array of shape (Nfreq, Npol, Npix) containing the observed
            multi-frequency data.
        covariance: ndarray
            Array of shape (Nfreq, Npol, Npix) containing the pixel covariance
            of the array `data`.
        fmatrix: object, `hoover.FMatrix`
            Instance of `hoover.FMatrix` that defines the component scaling
            in the fitted sky model.
        fixed_parameters: dict
            Dictionary, where key, value pairs correspond to parameter names
            and values to be fixed.
        priors: dict
            Dictionary, where key, value pairs correpsond to which parameters
            are allowed to vary, and the corresponding mean and standard
            deviation of the Gaussian prior.
        """
        if (data is not None) and (covariance is not None) and (frequencies is not None):
            self.data_setup(data, covariance, frequencies)

        if model is not None:
            self.model_setup(model)

    def __str__(self):
        """ Magic method for conveniently printing object summary.
        """
        msg = r"""
        Data
        ----
        Number of frequencies: {:d}
        Number of polarization channels: {:d}
        Number of pixels: {:d}

        Model
        -----
        Components: {:s}
        Free parameters: {:s} 
        Fixed parameters: {:s}
        """.format(self.nfreq, self.npol, self.npix, 
            " ".join(self._components),
            " ".join(self.free_parameters), 
            " ".join(self._fixed_parameters))
        return msg

    def __call__(self, theta, ret_neg=False):
        r""" This method computes the log probability at a point in parameter
        space specified by `pars` and any additional `kwargs` that may overwrite
        members of the `pars` dictionary.

        Parameters
        ----------
        theta: ndarray
            Array containing the parameters that are being varied.
            These should match the parameters given as priors when
            instantiating this object.
        ret_neg: bool (optional, default=False)
            If True, return the negative log likelihood. Else return
            the positive.

        Returns
        -------
        float
            The log probability at this point in parameter space.
        """
        try:
            assert len(theta) == len(self.free_parameters)
        except AssertionError:
            raise AssertionError("Must pass argument for each varied parameter.")
        lnprior = self._lnprior(theta)
        F = self._F(theta)
        N_T_inv = self._N_T_inv(theta, F=F)
        T_bar = self._T_bar(theta, F=F, N_T_inv=N_T_inv)
        lnP = _lnP(lnprior, T_bar, N_T_inv)
        # if ret_neg, return negative loglikelihood. Convenient for use
        # with minimization functions to find ML.
        if ret_neg:
            return - lnP
        return lnP

    def data_setup(self, data, covariance, frequencies):

        self.nfreq, self.npol, self.npix = data.shape
        self.frequencies = frequencies
        self.N_inv_d = (data, covariance)

    def model_setup(self, model):
        r""" Function to parse the model definition dictionary.

        The first level of the dictionary has keys corresponding
        to the different components, and must match one of the
        functions in `hoover.seds`. The values of the keys are
        dictionaries with `fixed` and `varied` keywords. These
        contain further dictionaries with fixed parameter 
        name / value pairs, and free parameter prior name / pairs.

        Parameters
        ----------
        model: dict
            Dictionary defining model.
        """
        self._components = list(model.keys())
        self._fmatrix = FMatrix(self._components)
        self._priors = {}
        self._fixed_parameters = {}
        for component in model.values():
            # get list of free parameters for this component
            prior = component.get('varied', None)
            # if there are not free parameters prior is None
            if prior is not None:
                self._priors.update(prior)
            # do the same for the fixed parameters
            fix_par = component.get('fixed', None)
            if fix_par is not None:
                self._fixed_parameters.update(fix_par)

        self.free_parameters = sorted(self._priors.keys())

    def load_data_from_hdf5(self, fpath, record):
        """ Method to instantiate data attributes using data saved
        in an HDF5 archive.

        Parameters
        ----------
        fpath: str, Path
            Path to the hdf5 archive.
        record: str
            Record in which the data is stored. This could be the
            name of the group in which the datasets are saved.
        """
        with h5py.File(fpath, 'r') as f:
            grp = f[record]
            data = grp['data'][...]
            cov = grp['cov'][...]
            frequencies = grp.attrs['frequencies']
        self.data_setup(data, cov, frequencies)

    def load_model_from_yaml(self, fpath):
        """ Method to load a model configuration from a yaml file. 
        """
        with open(fpath) as f:
            model = yaml.load(f, Loader=yaml.FullLoader)
        self.model_setup(model)

    def get_amplitude_expectation(self, theta):
        r""" Convenience function to return the component-separated expected
        amplitudes, `T_bar`, taking care of the relevant reshaping.
        """
        T_bar = self._T_bar(theta)
        return np.moveaxis(T_bar.reshape(self.npol, self.npix, -1), 2, 0)

    def get_amplitdue_covariance(self, *args, **kwargs):
        r""" Convenience function to return the component covariances,
        `N_T_inv`, for a given set of spectral parameters.
        """
        # NOT IMPLEMENTED
        #return self._N_T_inv(pars, *args, **kwargs)

    @property
    def free_parameters(self):
        return self.__free_parameters

    @free_parameters.setter
    def free_parameters(self, val):
        self.__free_parameters = val

    @property
    def N_inv(self):
        return self.__N_inv

    @N_inv.setter
    def N_inv(self, val):
        self.__N_inv = val

    @property
    def N_inv_d(self):
        return self.__N_inv_d

    @N_inv_d.setter
    def N_inv_d(self, val):
        """ Setter method for the inverse-variance weighted data.
        
        Parameters
        ----------
        val: tuple(ndarray)
            Tuple containing the data and covariance arrays. Arrays must have
            the same shape, and are expected in shape (Nfreq, Npol, Npix).
        """
        (data, cov) = val
        try:
            assert data.ndim == 3
            assert cov.ndim == 3
        except AssertionError:
            raise AssertionError("Data must have three dimensions, Nfreq, Npol, Npix")
        shape = [self.npix * self.npol, self.nfreq]
        data = _reorder_reshape_inputs(data, shape)
        self.N_inv = 1. / _reorder_reshape_inputs(cov, shape)
        self.__N_inv_d = data * self.N_inv

    def theta_0(self, npoints=None, seed=7837):
        """ Function to generate a starting guess for optimization
        or sampling by drawing a random point from the prior.

        Parameters
        ----------
        npoints: int (optional, default=None)
            If 

        Returns
        -------
        ndarray
            Array of length the number of free parameters.
        """
        key = jax.random.PRNGKey(seed)
        if npoints is not None:
            shape = (npoints, len(self.free_parameters))
        else:
            shape = (len(self.free_parameters),)
        out = jax.random.normal(key, shape=shape, dtype=np.float32)
        means = []
        stds = []
        for par in self.free_parameters:
            mean, std = self._priors[par]
            means.append(mean)
            stds.append(std)
        means = np.array(means)
        stds = np.array(stds)
        return means + out * stds 

    def _F(self, theta):
        free_parameters = dict(zip(self.free_parameters, theta))
        params = {**free_parameters, **self._fixed_parameters, 'nu': self.frequencies}
        return self._fmatrix(**params)

    def _N_T_inv(self, theta, F=None):
        if F is None:
            F = self._F(theta)
        return _N_T_inv(F, self.N_inv)

    def _T_bar(self, theta, F=None, N_T_inv=None):
        if F is None:
            F = self._F(theta)
        if N_T_inv is None:
            N_T_inv = self._N_T_inv(theta, F=F)
        return _T_bar(F, N_T_inv, self.N_inv_d)

    def _lnprior(self, theta):
        logprior = 0
        for arg, par in zip(theta, self.free_parameters):
            mean, std = self._priors[par]
            logprior += _log_gaussian(arg, mean, std)
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

def _reorder_reshape_inputs(arr, shape):
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
    return np.moveaxis(arr, (0, 1, 2), (2, 0, 1)).reshape(shape).astype(np.float32)