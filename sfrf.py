
import eagle as E
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


massBinLimits = np.linspace(-3.55, 3.25, 30)
massBins = np.logspace(-3.65, 3.15, 29)
print(massBinLimits)
print(np.log10(massBins))

import flares
from h5py_utilities import write_data_h5py, create_group_h5py, load_dict_from_hdf5, load_h5py

fl = flares.flares()

## Load GEAGLE ## 
fname = 'data/flares_%d.h5'
sfr = load_dict_from_hdf5(fname, 'sfr')
R = load_dict_from_hdf5(fname, 'radius')
vmask = load_dict_from_hdf5(fname, 'masks')


## Load Periodic ##
fname = 'data/periodic_%d.h5'
sfr_ref = load_dict_from_hdf5(fname, 'ref/sfr')
sfr_agn = load_dict_from_hdf5(fname, 'agn/sfr')


## Overdensity Weights ##
dat = np.loadtxt('../ic_selection/weights.txt', skiprows=1, delimiter=',')
weights = dat[:,8]
index = dat[:,1]

if len(weights) != len(fl.halos):
    print("Warning! number of weights not equal to number of halos")


#### Plot SFRF ####
fig, ax = plt.subplots(1,1,figsize=(6,6))

tag_idx = [10]
tag = fl.tags[10]
rtag = fl.ref_tags[5]

print(tag,rtag)
print(sfr.keys())
print(sfr_agn[rtag])
print(np.log10(sfr_agn[rtag]))

z = float(tag[5:].replace('p','.'))
scale_factor = 1. / (1+z)

phi_all = np.zeros(len(massBins))
hist_all = np.zeros(len(massBins))

for i,halo in enumerate(fl.halos):

    w = weights[np.where(["%04d"%i == halo for i in index])[0]]
    
    print(i,halo,w)

    sfr_temp = sfr[halo][tag][vmask[halo][tag]]

    V = (4./3) * np.pi * (R[halo][tag])**3
    if i == 0:
        V0 = V
    
    phi, phi_sigma, hist = fl.calc_df(sfr_temp, tag, V, massBinLimits) 

    V_factor = V0 / V

    phi_all += np.array(phi) * V_factor * w
    hist_all += hist


fl.plot_df(ax, phi_all, phi_sigma, hist_all, massBins, color='C0', label='G-EAGLE')

## Ref
phi, phi_sigma, hist = fl.calc_df(sfr_ref[rtag] * 1e10, rtag, 100**3, massBinLimits)
fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C1', label='Ref')

## AGN
phi, phi_sigma, hist = fl.calc_df(sfr_agn[rtag] * 1e10, rtag, 100**3, massBinLimits)
fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C2', label='AGNdT9')

ax.text(0.1, 0.1, '$z = %.1f$'%z, transform=ax.transAxes, size=12)
ax.set_xlim(8, 11.6)
ax.set_ylim(-6, -1.5) 
ax.grid(alpha=0.5)
    
ax.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)

ax.legend(frameon=False, loc=1);
plt.show()

# fig.savefig('output/gsmf.png', dpi=300, bbox_inches='tight')
