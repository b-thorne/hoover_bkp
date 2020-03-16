import jax.numpy as np
from jax import grad, jit, vmap


class FMatrix(object):
    """ Class to construct the foreground mixing matrix.

    This class models foreground SEDs of the different
    components of the sky model. This is an implementation of
    Equation (10) in 1608.00551, for a single set of
    spectral parameters.
    """
    def __init__(self, components):
        """ This method initializes the `FMatrix` with a list of
        function names. These functions must exist in the current
        global name space, and should be defined in this file.

        Parameters
        ----------
        components: list(str)
            List of function names. These functions, evaluated
            at a list of frequencies, will give the mixing
            matrix, `F`.
        """
        assert isinstance(components, list)
        # check that the list of components correspond to existing functions
        for component in components:
            assert component in ['dustmbb', 'syncpl', 'cmb', 'sync_curvedpl']
        self.components = components

    def __call__(self, *args, **parameters) -> np.ndarray:
        if parameters:
            self.parameters = parameters
        if args:
            self.parameters.update(*args)
        # evaluate each component function for the point in parameter space
        # specified by `parameters` N.B that each `comp_func` is passed all
        # the parameters - this requires that no two functions share argument
        # names.
        outputs = [globals()[comp_func](**self.parameters)[None] for comp_func in self.components]
        return np.concatenate(list(outputs))

@jit
def cmb(nu: np.ndarray, *args, **kwargs) -> np.ndarray:
    """ Function to compute CMB SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    """
    x = 0.0176086761 * nu
    ex = np.exp(x)
    sed = ex * (x / (ex - 1)) ** 2
    return sed


@jit
def syncpl(nu: np.ndarray, nu_ref_s: np.float32, beta_s: np.float32, *args, **kwargs) -> np.ndarray:
    """ Function to compute synchrotron power law SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    beta_s: float
        Power law index in RJ units.

    Returns
    -------
    array_like(float)
        Synchroton SED relative to reference frequency.
    """
    return (nu / nu_ref_s) ** beta_s


@jit
def sync_curvedpl(nu: np.ndarray, nu_ref_s: np.float32, beta_s: np.float32, beta_c: np.float32, *args, **kwargs) -> np.ndarray:
    """ Function to compute curved synchrotron power law SED.

    Parameters
    ----------
    nu: float, or array_like(float)
        Frequency in GHz.
    beta_s: float
        Power law index in RJ units.
    beta_c: float
        Power law index curvature.

    Returns
    -------
    array_like(float)
        Synchroton SED relative to reference frequency.
    """
    return (nu / nu_ref_s) ** (beta_s + beta_c * np.log(nu / nu_ref_s))


@jit
def dustmbb(nu: np.ndarray, nu_ref_d: np.float32, beta_d: np.float32, T_d: np.float32, *args, **kwargs) -> np.ndarray:
    """ Function to compute modified blackbody dust SED.

    Parameters
    ----------
    nu: float or array_like(float)
        Freuency at which to calculate SED.
    nu_ref_d: float
        Reference frequency in GHz.
    beta_d: float
        Power law index of dust opacity.
    T_d: float
        Temperature of the dust.

    Returns
    -------
    array_like(float)
        SED of dust modified black body relative to reference frequency.
    """
    x_to = np.float32(0.0479924466) * nu / T_d
    x_from = np.float32(0.0479924466) * nu_ref_d / T_d
    sed = (nu / nu_ref_d) ** (1. + beta_d) * (np.exp(x_from) - 1) / (np.exp(x_to) - 1)
    return sed
