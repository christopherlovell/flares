
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

massBinLimits = np.linspace(7.45, 13.25, 30)
massBins = np.logspace(7.55, 13.15, 29)
print(massBinLimits)
print(np.log10(massBins))

import flares
from h5py_utilities import write_data_h5py, create_group_h5py, load_dict_from_hdf5, load_h5py

fl = flares.flares()

## Load GEAGLE ## 
fname = 'data/flares_%d.h5'
mstar = load_dict_from_hdf5(fname, 'mstar')
centrals = load_dict_from_hdf5(fname, 'centrals')
sfr = load_dict_from_hdf5(fname, 'sfr')
R = load_dict_from_hdf5(fname, 'radius')
vmask = load_dict_from_hdf5(fname, 'masks')


# ## Load Periodic ##
# fname = 'data/periodic_%d.h5'
# mstar_ref = load_dict_from_hdf5(fname, 'ref/mstar')
# mstar_agn = load_dict_from_hdf5(fname, 'agn/mstar')


## Overdensity Weights ##
dat = np.loadtxt('../ic_selection/weights.txt', skiprows=1, delimiter=',')
weights = dat[:,8]
index = dat[:,1]


if len(weights) != len(fl.halos):
    print("Warning! number of weights not equal to number of halos")



tag_idx = [10]
tag = fl.tags[10]
rtag = fl.ref_tags[7]

print(tag,rtag)

z = float(tag[5:].replace('p','.'))
scale_factor = 1. / (1+z)

phi_sf = np.zeros(len(massBins))
hist_sf = np.zeros(len(massBins))

phi_q = np.zeros(len(massBins))
hist_q = np.zeros(len(massBins))

for i,halo in enumerate(fl.halos):

    w = weights[np.where(["%04d"%i == halo for i in index])[0]]
    
    print(i,halo,w)

    mstar_temp = mstar[halo][tag][vmask[halo][tag] & ~centrals[halo][tag]]
    sfr_temp = sfr[halo][tag][vmask[halo][tag] & ~centrals[halo][tag]]
    sfr_temp = sfr_temp[mstar_temp > 0.]
    mstar_temp = mstar_temp[mstar_temp > 0.] * 1e10
    
    mask = (mstar_temp > 0.)
    ssfr = sfr_temp[mask] / mstar_temp[mask]   

    V = (4./3) * np.pi * (R[halo][tag])**3
    if i == 0:
        V0 = V

    V_factor = V0 / V
    
    ## star -forming
    ssfr_lim = 5e-9
    mask = ssfr > ssfr_lim # np.log10(sfr_temp) > 0.
    phi, phi_sigma_sf, hist = fl.calc_df(mstar_temp[mask], tag, V, massBinLimits) 
    phi_sf += np.array(phi) * V_factor * w
    hist_sf += hist

    ## quiescent
    mask = ssfr < ssfr_lim #np.log10(sfr_temp) <= 0. 
    phi, phi_sigma_q, hist = fl.calc_df(mstar_temp[mask], tag, V, massBinLimits) 
    phi_q += np.array(phi) * V_factor * w
    hist_q += hist


#### Plot GSMF ####
fig, ax = plt.subplots(1,1,figsize=(6,6))
fl.plot_df(ax, phi_q, phi_sigma_q, hist_q, massBins=massBins, color='C1', label='G-EAGLE (quiescent)')
fl.plot_df(ax, phi_sf, phi_sigma_sf, hist_sf, massBins=massBins, color='C0', label='G-EAGLE (star-forming)')
fl.plot_df(ax, phi_sf+phi_q, phi_sigma_sf, hist_sf+hist_q, massBins=massBins, color='C2', label='G-EAGLE (composite)')

# #### Fit GSMF ####
# obs = [{'bin_edges': massBinLimits, 'N': hist_all, 'volume': V0}]
# 
# model = models.Schechter()
# 
# priors = {}
# priors['log10phi*'] = scipy.stats.uniform(loc = -6.0, scale = 6.0)
# priors['alpha'] = scipy.stats.uniform(loc = -3.0, scale = 4.0)
# priors['D*'] = scipy.stats.uniform(loc = 9., scale = 6.0)
# 
# fitter = fitDF.fitter(obs, model=model, priors=priors, output_directory='samples')
# samples = fitter.fit(nsamples=400, burn=100)
# 
# observations = [{'volume':V, 'sample': np.log10(mstar_temp), 'bin_edges': massBinLimits, 'N': hist_all}]
# 
# a = analyse.analyse(ID = 'samples', observations=observations)
# 
# fig = a.triangle(hist2d = True, ccolor='0.5')
# plt.savefig('images/posteriors.png')
# 
# fig = a.LF(observations=True,xlabel='$\mathrm{log_{10}}(M_{*} \,/\, M_{\odot})$')
# plt.savefig('images/fit_MF.png')

#### Periodic Volumes ####
# ## Ref
# phi, phi_sigma, hist = fl.calc_df(mstar_ref[rtag] * 1e10, rtag, 100**3, massBinLimits)
# fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C1', label='Ref')
# 
# ## AGN
# phi, phi_sigma, hist = fl.calc_df(mstar_agn[rtag] * 1e10, rtag, 50**3, massBinLimits)
# fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C2', label='AGNdT9')


# ax.text(0.1, 0.25, '$\mathrm{SFR} \,/\, M_{\odot} \, \mathrm{yr^{-1}} > 1$', 
#         transform=ax.transAxes, size=14)
ax.text(0.1, 0.2, '$\mathrm{sSFR} \,/\, \mathrm{yr^{-1}} > 5 \\times 10^{-9}$', 
        transform=ax.transAxes, size=14)
ax.text(0.1,0.3,'SATELLITES',size=14,transform=ax.transAxes)

ax.text(0.1, 0.1, '$z = %.1f$'%z, transform=ax.transAxes, size=12)
ax.set_xlim(8, 11.6)
ax.set_ylim(-6, -1) 
ax.grid(alpha=0.5)
    
ax.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)

ax.legend(frameon=False, loc=1);

# plt.show()
fig.savefig('images/gsmf_%.2f_sf_split_sat.png'%z, dpi=150, bbox_inches='tight')
