
import eagle as E
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


massBinLimits = np.linspace(7.45, 13.25, 30)
massBins = np.logspace(7.55, 13.15, 29)
print(massBinLimits)
print(np.log10(massBins))

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

if len(weights) != len(fl.halos):
    print("Warning! number of weights not equal to number of halos")


#### Plot GSMF ####

fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(12,8))

# gs = gridspec.GridSpec(8, 3)
# gs.update(wspace=0., hspace=0.)
# 
# ax1 = plt.subplot(gs[0:3, 0])
# ax2 = plt.subplot(gs[0:3, 1])
# ax3 = plt.subplot(gs[0:3, 2])
# ax4 = plt.subplot(gs[4:7, 0])
# ax5 = plt.subplot(gs[4:7, 1])
# ax6 = plt.subplot(gs[4:7, 2])

axes = [ax1,ax2,ax3,ax4,ax5,ax6]


for ax, tag, rtag in zip(axes, fl.tags[[5,6,7,8,9,10]], fl.ref_tags[[1,2,3,4,5,7]]): 
    
    phi_all = np.zeros(len(massBins))
    hist_all = np.zeros(len(massBins))

    for i,halo in enumerate(fl.halos):

        w = weights[np.where(["%04d"%i == halo for i in index])[0]]
        
        # print(i,halo,w)

        mstar_temp = mstar[halo][tag][vmask[halo][tag]]
        mstar_temp = mstar_temp[mstar_temp > 0.] * 1e10

        V = (4./3) * np.pi * (R[halo][tag])**3
        if i == 0:
            V0 = V
        
        phi, phi_sigma, hist = fl.calc_df(mstar_temp, tag, V, massBinLimits)

        V_factor = V0 / V

        phi_all += np.array(phi) * V_factor * w
        hist_all += hist
  
    
    fl.plot_df(ax, phi_all, phi_sigma, hist_all, massBins=massBins, color='C0', label='G-EAGLE')

    ## Ref
    phi, phi_sigma, hist = fl.calc_df(mstar_ref[rtag] * 1e10, rtag, 100**3, massBinLimits)
    fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C1', label='Ref')
    
    ## AGN
    phi, phi_sigma, hist = fl.calc_df(mstar_agn[rtag] * 1e10, rtag, 50**3, massBinLimits)
    fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C2', label='AGNdT9')
 

    z = float(tag[5:].replace('p','.'))
    ax.text(0.1, 0.2, '$z (resim) = %.2f$'%z, transform=ax.transAxes, size=12)
    z = float(rtag[5:].replace('p','.'))
    ax.text(0.1, 0.1, '$z (periodic) = %.2f$'%z, transform=ax.transAxes, size=12)
    
    ax.set_xlim(8, 11.6)
    ax.set_ylim(-6, -1.5) 
    ax.grid(alpha=0.5)
    

for ax in [ax2,ax3,ax5,ax6]:
    ax.set_yticklabels([])
    
ax1.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax4.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax5.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)
ax2.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)

ax1.legend(frameon=False, loc=1);
ax6.legend()

# plt.show()
fig.savefig('images/gsmf_multi.png', dpi=300, bbox_inches='tight')
