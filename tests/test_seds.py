import pytest
import jax.numpy as np
import numpy.testing as npt

from hoover.seds import cmb, syncpl, sync_curvedpl, dustmbb, FMatrix


def test_syncpl():
    assert syncpl(2., 1., 2.) == 4.


def test_sync_curvedpl():
    npt.assert_almost_equal(sync_curvedpl(np.exp(1), 1, 2, 1), np.exp(3), decimal=4)

def test_dustmbb():
    return

def test_fmatrix():
    sed = FMatrix(['dustmbb', 'syncpl', 'cmb'])
    parameters = {
        'nu': np.array([27., 39., 93., 145., 225., 280.]), 
        'nu_ref_d': np.float32(353), 
        'nu_ref_s': np.float32(23.), 
        'beta_d': np.float32(1.5), 
        'beta_s': np.float32(-3.), 
        'T_d': np.float32(20)
        }
    evalu = sed(**parameters)
    return
    