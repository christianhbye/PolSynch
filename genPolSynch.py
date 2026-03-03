# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "astropy>=7.2.0",
#     "healpy>=1.19.0",
#     "numpy>=2.4.2",
#     "scipy>=1.17.1",
# ]
# ///

"""
GENERATES POLARIZED SYNCHROTRON FOREGROUNDS AT DIFFERENT FREQUENCIES

"""

import numpy as np
import healpy as hp
import os
import sys
import random
from astropy.io import fits
import scipy.optimize
from scipy import stats
import invRM as iRM
import genRMmap as gRM


c = 299792458.0  # speed of light


def arcmin2rad(arcmin):
    return arcmin / 60 / 180 * np.pi


def jyb2Tb(jyb, res, freq, rmtf):
    """Convert Jy beam^-1 rmtf^-1 to Tb [K] given res in arcmin and freq in MHz."""
    rad = arcmin2rad(res)
    omega = rad ** 2
    return (jyb / omega * rmtf) * 1e-26 * (3 * 1e8) ** 2 / (2 * 1.38 * 1e-23 * (freq * 1e6) ** 2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_mwa', help='Path to DATA_MWA .npz file')
    parser.add_argument('--seed', dest='seed', default=1001, type=int,
                        help='seed for generating RM maps')
    parser.add_argument('--start', dest='start', default=50, type=float,
                        help='Starting frequency in MHz')
    parser.add_argument('--stop', dest='stop', default=200, type=float,
                        help='Stopping frequency in MHz')
    parser.add_argument('--nchan', dest='nchan', default=150, type=int,
                        help='Number of frequency channels')
    parser.add_argument('--nside', dest='nside', default=128, type=int,
                        help='Resolution of the map')
    parser.add_argument('--corr', dest='corr', action='store_true',
                        help='Option for generating RM correlations')
    parser.add_argument('--outdir', dest='outdir',
                        help='Path of Output map fits files')
    parser.add_argument('--storeRM', dest='storeRM', action='store_true',
                        help='Set option to store RM space Q,U maps')
    parser.add_argument('--readRM', dest='readRM', action='store_true',
                        help='Set option to use RM maps from fits file')
    parser.add_argument('--fileq', dest='fileq', default='mapRM_Q',
                        help='file RM Q maps from fits DEFAULT:mapRM_Q_nside[nside]_[RM].fits')
    parser.add_argument('--fileu', dest='fileu', default='mapRM_U',
                        help='file RM U maps from fits DEFAULT:mapRM_U_nside[nside]_[RM].fits')

    opts = parser.parse_args()

    print(opts)

    nside = int(opts.nside)
    npix = hp.nside2npix(nside)
    lmax = 3 * nside
    seed = int(opts.seed)
    outdir = str(opts.outdir)
    start, stop, nchan = float(opts.start), float(opts.stop), int(opts.nchan)
    RM = np.load(opts.data_mwa)['RM']
    sigma = np.load(opts.data_mwa)['sigma']
    alpha = np.load(opts.data_mwa)['alpha']
    rho_info = np.load(opts.data_mwa)['rho_info']
    frequency = np.linspace(start, stop, nchan)
    np.savez(outdir + 'input_info.npz', frequency=frequency, nside=nside, seed=seed, corr=opts.corr)

    # to convert in K
    convert = jyb2Tb(1, 15.6, 189, 4.3)  # See Bernardi et al 2013

    print("Input read")

    if opts.readRM:
        print("Reading RM maps from input files")
        RM_cube = np.zeros((len(RM), npix, 2), dtype=float)
        for rm in range(len(RM)):
            fileq = str(opts.fileq) + '_nside' + str(nside) + '_seed' + str(seed) + '_' + str(rm) + '.fits'
            fileu = str(opts.fileu) + '_nside' + str(nside) + '_seed' + str(seed) + '_' + str(rm) + '.fits'
            RM_cube[rm, :, 0] = hp.read_map(fileq)
            RM_cube[rm, :, 1] = hp.read_map(fileu)

            print("Q,U maps read for RM=", RM[rm])

    else:
        print("Generating RM full sky maps..")

        # create rho matrix nRMxlmax
        rho = gRM.genRho(len(RM), lmax, rho_info, opts.corr)
        # create cov matrix (lmax, nRM, nRM)
        cov_vec = np.zeros((lmax, len(RM), len(RM)), dtype=float)
        for l in range(1, lmax):
            cov_vec[l, :, :] = gRM.RM_cov(len(RM), l, alpha, rho)

        # generate the full RM cube
        RM_cube = np.zeros((len(RM), npix, 2), dtype=float)
        RM_cube = gRM.genRMcube_corr(len(RM), nside, seed, cov_vec, sigma)
        if opts.storeRM:
            print("Storing RM maps")
            for rm in range(len(RM)):
                fname_Q = outdir + 'mapRM_Q_nside' + str(nside) + '_seed' + str(seed) + '_' + str(rm) + '.fits'
                hp.write_map(fname_Q, RM_cube[rm, :, 0])
                fname_U = outdir + 'mapRM_U_nside' + str(nside) + '_seed' + str(seed) + '_' + str(rm) + '.fits'
                hp.write_map(fname_U, RM_cube[rm, :, 1])

        print("RM full sky maps generated.")

    print("Apply inverse RM")

    q_maps = np.zeros((len(frequency), npix), dtype=float)
    u_maps = np.zeros((len(frequency), npix), dtype=float)
    for ii, ff in enumerate(frequency):
        p_map = iRM.P_l2(npix, RM_cube[:, :, 0], RM_cube[:, :, 1], RM, ff)
        print("Computed freq=", ii, ff)

        map_Q = p_map.real * convert
        map_U = p_map.imag * convert

        q_maps[ii, :] = map_Q
        u_maps[ii, :] = map_U

    print("Storing output maps")
    np.savez(outdir + 'pol_synch_maps.npz', q_maps=q_maps, u_maps=u_maps, frequency=frequency, seed=seed, nside=nside)
