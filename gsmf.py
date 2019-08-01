
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


if len(weights) != len(fl.halos):
    print("Warning! number of weights not equal to number of halos")



tag_idx = [10]
tag = fl.tags[10]
rtag = fl.ref_tags[7]

print(tag,rtag)

z = float(tag[5:].replace('p','.'))
scale_factor = 1. / (1+z)

phi_all = np.zeros(len(massBins))
hist_all = np.zeros(len(massBins))

for i,halo in enumerate(fl.halos):

    w = weights[np.where(["%04d"%i == halo for i in index])[0]]
    
    print(i,halo,w)

    mstar_temp = mstar[halo][tag][vmask[halo][tag]]
    mstar_temp = mstar_temp[mstar_temp > 0.] * 1e10

    V = (4./3) * np.pi * (R[halo][tag])**3
    if i == 0:
        V0 = V
    
    phi, phi_sigma, hist = fl.calc_df(mstar_temp, tag, V, massBinLimits) 

    V_factor = V0 / V

    phi_all += np.array(phi) * V_factor * w
    hist_all += hist


#### Plot GSMF ####
fig, ax = plt.subplots(1,1,figsize=(6,6))
fl.plot_df(ax, phi_all, phi_sigma, hist_all, massBins=massBins, color='C0', label='G-EAGLE')

ax.text(0.1, 0.1, '$z = %.1f$'%z, transform=ax.transAxes, size=12)
ax.set_xlim(8, 11.6)
ax.set_ylim(-6, -1.5) 
ax.grid(alpha=0.5)
    
ax.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)

ax.legend(frameon=False, loc=1);

# plt.show()
fig.savefig('images/gsmf_%.2f.png'%z, dpi=150, bbox_inches='tight')



#### Fit GSMF ####
obs = [{'bin_edges': massBinLimits, 'N': hist_all, 'volume': V0}]

model = models.DoubleSchechter()

priors = {}
priors['log10phi*_1'] = scipy.stats.uniform(loc = -6, scale = 7.0)
priors['alpha_1'] = scipy.stats.uniform(loc = -3.0, scale = 3.0)
priors['log10phi*_2'] = scipy.stats.uniform(loc = -6, scale = 7.0)
priors['alpha_2'] = scipy.stats.uniform(loc = -3.0, scale = 3.0)
priors['D*'] = scipy.stats.uniform(loc = 9., scale = 7.0)

fitter = fitDF.fitter(obs, model=model, priors=priors, output_directory='samples')
samples = fitter.fit(nsamples=800, burn=400)

observations = [{'volume':V, 'sample': np.log10(mstar_temp), 'bin_edges': massBinLimits, 'N': hist_all}]

a = analyse.analyse(ID = 'samples', model=model, observations=observations)

fig = a.triangle(hist2d = True, ccolor='0.5')
plt.savefig('images/posteriors.png')

fig = a.LF(observations=True,xlabel='$\mathrm{log_{10}}(M_{*} \,/\, M_{\odot})$')
plt.savefig('images/fit_MF.png')

# ## Ref
# phi, phi_sigma, hist = fl.calc_df(mstar_ref[rtag] * 1e10, rtag, 100**3, massBinLimits)
# fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C1', label='Ref')
# 
# ## AGN
# phi, phi_sigma, hist = fl.calc_df(mstar_agn[rtag] * 1e10, rtag, 50**3, massBinLimits)
# fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C2', label='AGNdT9')


