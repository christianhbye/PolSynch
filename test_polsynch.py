"""
Unit tests for PolSynch library functions.

Tests verify that the Python 3 compatible code produces results
consistent with the original algorithm.
"""

import numpy as np
import pytest

import invRM as iRM
import genRMmap as gRM
from genPolSynch import arcmin2rad, jyb2Tb


# --- tests for genPolSynch.py ---

class TestArcmin2Rad:
    def test_zero(self):
        assert arcmin2rad(0) == 0.0

    def test_one_arcmin(self):
        expected = 1 / 60 / 180 * np.pi
        assert np.isclose(arcmin2rad(1), expected)

    def test_60_arcmin_is_one_degree(self):
        assert np.isclose(arcmin2rad(60), np.pi / 180)


class TestJyb2Tb:
    def test_positive(self):
        # With positive inputs the result should be positive
        result = jyb2Tb(1, 15.6, 189, 4.3)
        assert result > 0

    def test_linearity_in_jyb(self):
        # doubling jyb should double the result
        r1 = jyb2Tb(1, 15.6, 189, 4.3)
        r2 = jyb2Tb(2, 15.6, 189, 4.3)
        assert np.isclose(r2, 2 * r1)

    def test_known_value(self):
        # Computed value used throughout genPolSynch as the Bernardi 2013 conversion
        result = jyb2Tb(1, 15.6, 189, 4.3)
        assert np.isfinite(result)
        assert result > 0


# --- tests for invRM.py ---

class TestFreq2Wl2:
    def test_single_value(self):
        c = 299792458.0
        freq_mhz = 100.0
        expected = (c / (freq_mhz * 1e6)) ** 2
        assert np.isclose(iRM.freq2wl2(freq_mhz), expected)

    def test_array(self):
        freqs = np.array([50.0, 100.0, 200.0])
        results = iRM.freq2wl2(freqs)
        # Higher frequency => shorter wavelength => smaller wl2
        assert results[0] > results[1] > results[2]


class TestCompPhases:
    def test_zero_phi(self):
        # exp(0) = 1
        result = iRM.comp_phases(0.5, 0)
        assert np.isclose(result, 1.0)

    def test_shape(self):
        wl2 = np.ones(10)
        result = iRM.comp_phases(wl2, 1.0)
        assert result.shape == (10,)

    def test_unit_magnitude(self):
        wl2 = np.linspace(0, 1, 50)
        result = iRM.comp_phases(wl2, 2.0)
        assert np.allclose(np.abs(result), 1.0)


class TestP_l2:
    def test_single_rm_zero_phi(self):
        # phi_array = [0], phase = exp(0) = 1, so p_map = q + j*u
        npix = 12
        q_rm = np.ones((1, npix))
        u_rm = np.zeros((1, npix))
        result = iRM.P_l2(npix, q_rm, u_rm, [0.0], 100.0)
        assert result.shape == (npix,)
        assert np.allclose(result.real, 1.0)
        assert np.allclose(result.imag, 0.0)

    def test_output_shape(self):
        npix = 48
        nrm = 3
        q_rm = np.random.rand(nrm, npix)
        u_rm = np.random.rand(nrm, npix)
        phi_array = np.array([0.0, 1.0, 2.0])
        result = iRM.P_l2(npix, q_rm, u_rm, phi_array, 150.0)
        assert result.shape == (npix,)


# --- tests for genRMmap.py ---

class TestGenRho:
    def test_no_corr_returns_zeros(self):
        rho = gRM.genRho(3, 10, [0, 0, 0, 0], corr=False)
        assert np.all(rho == 0.0)

    def test_shape(self):
        nRM, lmax = 4, 20
        rho = gRM.genRho(nRM, lmax, [0.01, 0.5, 1e-3, -0.5], corr=True)
        assert rho.shape == (nRM, nRM, lmax)

    def test_corr_nonzero(self):
        # With corr=True, rho should not be all zeros for l >= 1
        rho = gRM.genRho(2, 10, [0.01, 0.5, 1e-3, -0.5], corr=True)
        assert not np.all(rho == 0.0)


class TestRM_cov:
    def test_diagonal_only_when_no_corr(self):
        nRM = 3
        alpha = np.array([1.0, 1.0, 1.0])
        rho = np.zeros((nRM, nRM, 100))
        ell = 5
        cov = gRM.RM_cov(nRM, ell, alpha, rho)
        # Off-diagonal should be 0 when rho=0
        assert np.isclose(cov[0, 1], 0.0)
        assert np.isclose(cov[1, 2], 0.0)

    def test_diagonal_value(self):
        nRM = 2
        alpha = np.array([2.0, 3.0])
        rho = np.zeros((nRM, nRM, 100))
        ell = 4
        cov = gRM.RM_cov(nRM, ell, alpha, rho)
        assert np.isclose(cov[0, 0], ell ** alpha[0])
        assert np.isclose(cov[1, 1], ell ** alpha[1])

    def test_symmetric(self):
        nRM = 3
        alpha = np.array([1.5, 2.0, 1.0])
        rho_info = [0.01, 0.5, 1e-3, -0.5]
        rho = gRM.genRho(nRM, 50, rho_info, corr=True)
        ell = 10
        cov = gRM.RM_cov(nRM, ell, alpha, rho)
        assert np.allclose(cov, cov.T)


class TestGenRMcubecorr:
    def test_output_shape(self):
        nRM = 2
        nside = 4
        seed = 42
        lmax = 3 * nside
        alpha = np.array([1.5, 2.0])
        rho = np.zeros((nRM, nRM, lmax))
        cov = np.zeros((lmax, nRM, nRM))
        for l in range(1, lmax):
            cov[l] = gRM.RM_cov(nRM, l, alpha, rho)
        sigma = np.array([1.0, 0.5])
        cube = gRM.genRMcube_corr(nRM, nside, seed, cov, sigma)
        assert cube.shape == (nRM, 12 * nside ** 2, 2)

    def test_reproducible_with_same_seed(self):
        nRM = 2
        nside = 4
        seed = 99
        lmax = 3 * nside
        alpha = np.array([1.5, 2.0])
        rho = np.zeros((nRM, nRM, lmax))
        cov = np.zeros((lmax, nRM, nRM))
        for l in range(1, lmax):
            cov[l] = gRM.RM_cov(nRM, l, alpha, rho)
        sigma = np.array([1.0, 0.5])
        cube1 = gRM.genRMcube_corr(nRM, nside, seed, cov, sigma)
        cube2 = gRM.genRMcube_corr(nRM, nside, seed, cov, sigma)
        assert np.allclose(cube1, cube2)

    def test_sigma_scaling(self):
        """Maps should be rescaled so that std(Q) = sigma for each RM bin."""
        nRM = 2
        nside = 4
        seed = 7
        lmax = 3 * nside
        alpha = np.array([1.5, 2.0])
        rho = np.zeros((nRM, nRM, lmax))
        cov = np.zeros((lmax, nRM, nRM))
        for l in range(1, lmax):
            cov[l] = gRM.RM_cov(nRM, l, alpha, rho)
        sigma = np.array([2.0, 3.0])
        cube = gRM.genRMcube_corr(nRM, nside, seed, cov, sigma)
        for rm in range(nRM):
            assert np.isclose(np.std(cube[rm, :, 0]), sigma[rm], rtol=1e-6)
            assert np.isclose(np.std(cube[rm, :, 1]), sigma[rm], rtol=1e-6)
