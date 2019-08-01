
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
R = load_dict_from_hdf5(fname, 'radius')
vmask = load_dict_from_hdf5(fname, 'masks')


## Load Periodic ##
fname = 'data/periodic_%d.h5'
mstar_ref = load_dict_from_hdf5(fname, 'ref/mstar')
centrals_ref = load_dict_from_hdf5(fname, 'ref/centrals')
mstar_agn = load_dict_from_hdf5(fname, 'agn/mstar')
centrals_agn = load_dict_from_hdf5(fname, 'agn/centrals')


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

phi_c = np.zeros(len(massBins))
hist_c = np.zeros(len(massBins))

phi_s = np.zeros(len(massBins))
hist_s = np.zeros(len(massBins))

for i,halo in enumerate(fl.halos):

    w = weights[np.where(["%04d"%i == halo for i in index])[0]]
    
    V = (4./3) * np.pi * (R[halo][tag])**3
    if i == 0:
        V0 = V

    V_factor = V0 / V
    
    print(i,halo,w)

    mstar_temp = mstar[halo][tag][vmask[halo][tag] & centrals[halo][tag]]
    mstar_temp = mstar_temp[mstar_temp > 0.] * 1e10
    
    phi, phi_sigma, hist = fl.calc_df(mstar_temp, tag, V, massBinLimits) 
    phi_c += np.array(phi) * V_factor * w
    hist_c += hist
    
    mstar_temp = mstar[halo][tag][vmask[halo][tag] & ~centrals[halo][tag]]
    mstar_temp = mstar_temp[mstar_temp > 0.] * 1e10
    
    phi, phi_sigma, hist = fl.calc_df(mstar_temp, tag, V, massBinLimits) 
    phi_s += np.array(phi) * V_factor * w
    hist_s += hist



#### Plot GSMF ####
fig, ax = plt.subplots(1,1,figsize=(6,6))
fl.plot_df(ax, phi_c, phi_sigma, hist_c, massBins=massBins, color='red', label='G-EAGLE (cen)')
fl.plot_df(ax, phi_s, phi_sigma, hist_s, massBins=massBins, color='darkred', label='G-EAGLE (sat)')


#### Periodic Volumes ####
## Ref
phi, phi_sigma, hist = fl.calc_df(mstar_ref[rtag][centrals_ref[rtag]] * 1e10, 
                                  rtag, 100**3, massBinLimits)
fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='g', label='Ref (cen)')

phi, phi_sigma, hist = fl.calc_df(mstar_ref[rtag][~centrals_ref[rtag]] * 1e10, 
                                  rtag, 100**3, massBinLimits)
fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='mediumseagreen', label='Ref (sat)')

## AGN
phi, phi_sigma, hist = fl.calc_df(mstar_agn[rtag][centrals_agn[rtag]] * 1e10, 
                                  rtag, 50**3, massBinLimits)
fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='m', label='AGNdT9 (cen)')

phi, phi_sigma, hist = fl.calc_df(mstar_agn[rtag][~centrals_agn[rtag]] * 1e10, 
                                  rtag, 50**3, massBinLimits)
fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='hotpink', label='AGNdT9 (sat)')


# ax.text(0.1, 0.25, '$\mathrm{SFR} \,/\, M_{\odot} \, \mathrm{yr^{-1}} > 1$', 
#         transform=ax.transAxes, size=14)
ax.text(0.1, 0.2, '$\mathrm{sSFR} \,/\, \mathrm{yr^{-1}} > 5 \\times 10^{-9}$', 
        transform=ax.transAxes, size=14)

ax.text(0.1, 0.1, '$z = %.1f$'%z, transform=ax.transAxes, size=14)
ax.set_xlim(8, 11.6)
ax.set_ylim(-6, -1) 
ax.grid(alpha=0.5)
    
ax.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)

ax.legend(frameon=False, loc=1);

# plt.show()
fig.savefig('images/gsmf_%.2f_cen_sat.png'%z, dpi=150, bbox_inches='tight')
