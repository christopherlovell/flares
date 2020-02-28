"""

    All the functions listed here requires the generation of the particle
    information file.

"""

import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
from functools import partial
import schwimmbad

import SynthObs
from SynthObs.SED import models

import FLARE
import FLARE.filters
from FLARE.photom import lum_to_M, M_to_lum

import h5py

def DTM_fit(Z, Age):

    """
    Fit function from L-GALAXIES dust modeling
    Formula uses Age in Gyr while the supplied Age is in Myr
    """

    D0, D1, alpha, beta, gamma = 0.008, 0.329, 0.017, -1.337, 2.122
    tau = 5e-5/(D0*Z)
    DTM = D0 + (D1-D0)*(1.-np.exp(-alpha*(Z**beta)*((Age/(1e3*tau))**gamma)))
    if np.isnan(DTM) or np.isinf(DTM): DTM = 0.

    return DTM


def get_data(ii, tag, inp = 'FLARES'):

    num = str(ii)
    if inp == 'FLARES':
        if len(num) == 1:
            num =  '0'+num

        sim = rF"./data/FLARES_{num}_sp_info.hdf5"

    else:
        sim = rF"./data/EAGLE_{inp}_sp_info.hdf5"

    with h5py.File(sim, 'r') as hf:
        S_len = np.array(hf[tag+'/Galaxy'].get('S_Length'), dtype = np.int64)
        G_len = np.array(hf[tag+'/Galaxy'].get('G_Length'), dtype = np.int64)
        S_mass = np.array(hf[tag+'/Particle'].get('S_MassInitial'), dtype = np.float64)
        S_Z = np.array(hf[tag+'/Particle'].get('S_Z'), dtype = np.float64)
        S_age = np.array(hf[tag+'/Particle'].get('S_Age'), dtype = np.float64)*1e3
        S_los = np.array(hf[tag+'/Particle'].get('S_los'), dtype = np.float64)
        G_Z = np.array(hf[tag+'/Particle'].get('G_Z'), dtype = np.float64)

    begin = np.zeros(len(S_len), dtype = np.int64)
    end = np.zeros(len(S_len), dtype = np.int64)
    begin[1:] = np.cumsum(S_len)[:-1]
    end = np.cumsum(S_len)

    gbegin = np.zeros(len(G_len), dtype = np.int64)
    gend = np.zeros(len(G_len), dtype = np.int64)
    gbegin[1:] = np.cumsum(G_len)[:-1]
    gend = np.cumsum(G_len)

    return S_mass, S_Z, S_age, S_los, G_Z, S_len, G_len, begin, end, gbegin, gend


def lum(sim, kappa, tag, inp = 'FLARES', IMF = 'Chabrier_300', LF = True, filters = ['FAKE.TH.FUV'], Type = 'Total'):

    S_mass, S_Z, S_age, S_los, G_Z, S_len, G_len, begin, end, gbegin, gend = get_data(sim, tag, inp)

    if np.isscalar(filters):
        Lums = np.zeros(len(begin), dtype = np.float64)
    else:
        Lums = np.zeros((len(begin), len(filters)), dtype = np.float64)

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    model.dust = {'slope': -1.0} # define dust curve

    # --- create rest-frame luminosities
    F = FLARE.filters.add_filters(filters, new_lam = model.lam)
    model.create_Lnu_grid(F) # --- create new L grid for each filter. In units of erg/s/Hz


    for jj in range(len(begin)):

        Masses = S_mass[begin[jj]:end[jj]]
        Ages = S_age[begin[jj]:end[jj]]
        Metallicities = S_Z[begin[jj]:end[jj]]
        MetSurfaceDensities = S_los[begin[jj]:end[jj]]

        # GMetallicities = G_Z[gbegin[jj]:gend[jj]]
        # Mage = np.nansum(Masses*Ages)/np.nansum(Masses)
        # Z = np.nanmean(GMetallicities)
        # if kappa == 0:
        #     tauVs = kappa * MetSurfaceDensities
        # else:
        #     tauVs = DTM_fit(Z, Mage) * MetSurfaceDensities

        if Type == 'Total':
            tauVs = kappa * MetSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
            fesc = 0.0

        elif Type == 'Pure-stellar':
            tauVs = np.zeros(len(Masses))
            fesc = 1.0

        elif Type == 'Only-BC':
            tauVs = np.zeros(len(Masses))
            fesc = 0.0

        else:
            ValueError(F"Undefined Type {Type}")

        Lnu = models.generate_Lnu(model, Masses, Ages, Metallicities, tauVs, F, fesc = fesc) # --- calculate rest-frame Luminosity. In units of erg/s/Hz
        Lums[jj] = list(Lnu.values())

    return Lums#, S_len + G_len



def flux(sim, kappa, tag, inp = 'FLARES', IMF = 'Chabrier_300', filters = FLARE.filters.NIRCam_W, Type = 'Total'):

    S_mass, S_Z, S_age, S_los, G_Z, S_len, G_len, begin, end, gbegin, gend = get_data(sim, tag, inp)

    if np.isscalar(filters):
        Fnus = np.zeros(len(begin), dtype = np.float64)
    else:
        Fnus = np.zeros((len(begin), len(filters)), dtype = np.float64)

    model = models.define_model(F'BPASSv2.2.1.binary/{IMF}') # DEFINE SED GRID -
    model.dust = {'slope': -1.0} # define dust curve

    z = float(tag[5:].replace('p','.'))
    F = FLARE.filters.add_filters(filters, new_lam = model.lam * (1. + z))

    cosmo = FLARE.default_cosmo()

    model.create_Fnu_grid(F, z, cosmo) # --- create new Fnu grid for each filter. In units of nJy/M_sol

    for jj in range(len(begin)):
        Masses = S_mass[begin[jj]:end[jj]]
        Ages = S_age[begin[jj]:end[jj]]
        Metallicities = S_Z[begin[jj]:end[jj]]
        MetSurfaceDensities = S_los[begin[jj]:end[jj]]

        # GMetallicities = G_Z[gbegin[jj]:gend[jj]]
        #
        # Mage = Masses*Ages/np.nansum(Masses)
        # Z = np.nanmean(GMetallicities)
        # if kappa == 0:
        #     tauVs = kappa * MetSurfaceDensities
        # else:
        #     tauVs = DTM_fit(Z, Mage) * MetSurfaceDensities

        if Type == 'Total':
            tauVs = kappa * MetSurfaceDensities # --- calculate V-band (550nm) optical depth for each star particle
            fesc = 0.0
        elif Type == 'Pure-stellar':
            tauVs = np.zeros(len(Masses))
            fesc = 1.0
        elif Type == 'Only-BC':
            tauVs = np.zeros(len(Masses))
            fesc = 0.0
        else:
            ValueError(F"Undefined Type {Type}")

        Fnu = models.generate_Fnu(model, Masses, Ages, Metallicities, tauVs, F, fesc = fesc) # --- calculate rest-frame flux of each object in nJy

        Fnus[jj] = list(Fnu.values())

    return Fnus


def get_lum(sim, kappa, tag, IMF = 'Chabrier_300', bins = np.arange(-24, -16, 0.5), inp = 'FLARES', LF = True, filters = ['FAKE.TH.FUV'], Type = 'Total'):

    try:
        Lums = lum(sim, kappa, tag, IMF=IMF, inp=inp, LF=LF, filters=filters, Type = Type)

    except Exception as e:
        Lums = np.ones(len(filters))*np.nan
        print (e)


    if LF:
        tmp, edges = np.histogram(lum_to_M(Lums), bins = bins)
        return tmp

    else:
        return Lums



def get_lum_all(kappa, tag, IMF = 'Chabrier_300', bins = np.arange(-24, -16, 0.5), inp = 'FLARES', LF = True, filters = ['FAKE.TH.FUV'], Type = 'Total'):

    print (f"Getting luminosities for tag {tag} with kappa = {kappa}")

    if inp == 'FLARES':
        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0,len(weights))

        calc = partial(get_lum, kappa = kappa, tag = tag, IMF = IMF, bins = bins, inp = inp, LF = LF, filters = filters, Type = Type)

        pool = schwimmbad.MultiPool(processes=12)
        dat = np.array(list(pool.map(calc, sims)))
        pool.close()

        if LF:
            hist = np.sum(dat, axis = 0)
            out = np.zeros(len(bins)-1)
            err = np.zeros(len(bins)-1)
            for ii, sim in enumerate(sims):
                err+=np.square(np.sqrt(dat[ii])*weights[ii])
                out+=dat[ii]*weights[ii]

            return out, hist, np.sqrt(err)

        else:
            return dat

    else:
        out = get_lum(00, kappa = kappa, tag = tag, IMF = IMF, bins = bins, inp = inp, LF = LF, filters = filters, Type = Type)

        return out


def get_flux(sim, kappa, tag, IMF = 'Chabrier_300', inp = 'FLARES', filters = FLARE.filters.NIRCam, Type = 'Total'):

    try:
        Fnus = flux(sim, kappa, tag, IMF=IMF, inp=inp, filters=filters, Type = Type)

    except Exception as e:
        Fnus = np.ones(len(filters))*np.nan
        print (e)

    return Fnus

def get_flux_all(kappa, tag, IMF = 'Chabrier_300', inp = 'FLARES', filters = FLARE.filters.NIRCam, Type = 'Total'):

    print (f"Getting fluxes for tag {tag} with kappa = {kappa}")

    if inp == 'FLARES':

        df = pd.read_csv('weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0,len(weights))

        calc = partial(get_flux, kappa = kappa, tag = tag, IMF = IMF, inp = inp, filters = filters, Type = Type)

        pool = schwimmbad.MultiPool(processes=12)
        out = np.array(list(pool.map(calc, sims)))
        pool.close()

    else:

        out = get_flux(00, kappa = kappa, tag = tag, IMF = IMF, inp = inp, filters = filters, Type = Type)

    return out
