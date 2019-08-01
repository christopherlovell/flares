
import eagle as E
import numpy as np
import scipy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
mpl.rcParams['text.usetex'] = True

import fitDF.fitDF as fitDF
import fitDF.models as models
import fitDF.analyse as analyse

massBinLimits = np.linspace(7.4, 13.4, 21)
massBins = np.logspace(7.55, 13.25, 20)
# print(massBinLimits)
# print(np.log10(massBins))

import flares
from h5py_utilities import write_data_h5py, create_group_h5py, load_dict_from_hdf5

fl = flares.flares()

## Load GEAGLE ## 
fname = 'data/flares_%d.h5'
mstar = load_dict_from_hdf5(fname, 'mstar')
R = load_dict_from_hdf5(fname, 'radius')
vmask = load_dict_from_hdf5(fname, 'masks')

## Load Periodic ##
fname = 'data/periodic_%d.h5'
mstar_ref = load_dict_from_hdf5(fname, 'ref/mstar')
mstar_agn = load_dict_from_hdf5(fname, 'agn/mstar')

## Overdensity Weights ##
dat = np.loadtxt('../ic_selection/weights.txt', skiprows=1, delimiter=',')
weights = dat[:,8]
index = dat[:,1]

# if len(weights) != len(fl.halos):
#     print("Warning! number of weights not equal to number of halos")


def fitdf(hist_all, V, mstar_temp, name):

    obs = [{'bin_edges': massBinLimits, 'N': hist_all, 'volume': V}] 
    model = models.DoubleSchechter()
 
    priors = {}
    priors['log10phi*_1'] = scipy.stats.uniform(loc = -6.0, scale = 6.0)
    priors['alpha_1'] = scipy.stats.uniform(loc = -3.0, scale = 4.0)
    priors['log10phi*_2'] = scipy.stats.uniform(loc = -6.0, scale = 6.0)
    priors['alpha_2'] = scipy.stats.uniform(loc = -3.0, scale = 4.0)
    priors['D*'] = scipy.stats.uniform(loc = 9., scale = 6.0)
 
    fitter = fitDF.fitter(obs, model=model, priors=priors, output_directory='samples')
    samples = fitter.fit(nsamples=400, burn=100, sample_save_ID=name)
 
    observations = [{'volume':V, 'sample': np.log10(mstar_temp), 'bin_edges': massBinLimits, 'N': hist_all}]
    a = analyse.analyse(ID='samples', model=model, sample_save_ID=name, observations=observations)
 
    fig = a.triangle(hist2d = True, ccolor='0.5')
    plt.savefig('images/%s_posteriors.png'%name)
 
    fig = a.LF(observations=True,xlabel='$\mathrm{log_{10}}(M_{*} \,/\, M_{\odot})$')
    plt.savefig('images/%s_fit_MF.png'%name)
 


def fit(tags): 
    
    tag, rtag = tags
    print(tag)

    phi_all = np.zeros(len(massBins))
    hist_all = np.zeros(len(massBins))

    for i,halo in enumerate(fl.halos):

        w = weights[np.where(["%04d"%i == halo for i in index])[0]]
        
        mstar_temp = mstar[halo][tag][vmask[halo][tag]]
        mstar_temp = mstar_temp[mstar_temp > 0.] * 1e10

        V = (4./3) * np.pi * (R[halo][tag])**3
        if i == 0:
            V0 = V
        
        phi, phi_sigma, hist = fl.calc_df(mstar_temp, tag, V, massBinLimits)

        V_factor = V0 / V

        phi_all += np.array(phi) * V_factor * w
        hist_all += hist
  
    
    fitdf(hist_all, V0, mstar_temp, name='geagle_gsmf_%s'%tag)
    
    ## Ref ##
    phi, phi_sigma, hist = fl.calc_df(mstar_ref[rtag] * 1e10, rtag, 100**3, massBinLimits)
    fitdf(hist, 100**3, mstar_ref[rtag] * 1e10, name='ref_gsmf_%s'%rtag)

    
    ## AGN ##
    phi, phi_sigma, hist = fl.calc_df(mstar_agn[rtag] * 1e10, rtag, 50**3, massBinLimits)
    fitdf(hist, 100**3, mstar_agn[rtag] * 1e10, name='agn_gsmf_%s'%rtag)
    
    return None




from schwimmbad import MultiPool
from functools import partial

with MultiPool() as pool:
    tags = list(zip(fl.tags[[5,6,7,8,9,10]], fl.ref_tags[[1,2,3,4,5,7]]))
    values = list(pool.map(fit, tags))


