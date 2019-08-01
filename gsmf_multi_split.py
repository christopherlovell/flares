
import eagle as E
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
mpl.rcParams['text.usetex'] = True


massBinLimits = np.linspace(7.4, 13.4, 21)
massBins = np.logspace(7.55, 13.25, 20)
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
index = dat[:,1]
sigma = dat[:,7]
print(index,sigma,fl.halos)

## reverse sort..
sigma = sigma[::-1]
index = index[::-1]

# hmask = np.hstack([np.where(["%04d"%i == halo for halo in fl.halos])[0] for i in index]) 
# print(hmask)

s_range = np.arange(-3,6)
ticks = np.linspace(0.05, .95, len(s_range))
colors = [ cm.viridis(i) for i in ticks ]

# if len(weights) != len(fl.halos):
#     print("Warning! number of weights not equal to number of halos")


#### Plot GSMF ####

fig = plt.figure(figsize=(12,8))

gs = gridspec.GridSpec(2, 4, width_ratios=[1,1,1,0.06])
gs.update(wspace=0., hspace=0.)

ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])
ax3 = plt.subplot(gs[0, 2])
ax4 = plt.subplot(gs[1, 0])
ax5 = plt.subplot(gs[1, 1])
ax6 = plt.subplot(gs[1, 2])
cax = plt.subplot(gs[0,3])
axes = [ax1,ax2,ax3,ax4,ax5,ax6]

for k, (ax, tag, rtag) in enumerate(zip(axes, fl.tags[[5,6,7,8,9,10]], fl.ref_tags[[1,2,3,4,5,7]])): 
    
    for j,(sig,C) in enumerate(zip(s_range,colors)):
        
        phi_all = np.zeros(len(massBins))
        hist_all = np.zeros(len(massBins))

        mask = (sigma < sig + 0.5) & (sigma > sig - 0.5)

        if np.sum(mask) > 0:

            # find total volume
            V = np.sum([((4./3) * np.pi * (R[halo][tag])**3) for halo in fl.halos[mask]])
            
            for i,halo in enumerate(fl.halos[mask]):
                if (k==0): print(k,tag,i,halo,j,sig)
    
                mstar_temp = mstar[halo][tag][vmask[halo][tag]]
                mstar_temp = mstar_temp[mstar_temp > 0.] * 1e10
                
                phi, phi_sigma, hist = fl.calc_df(mstar_temp, tag, V, massBinLimits)
        
                phi_all += np.array(phi) 
                hist_all += hist

            if np.sum(phi_all > 0) > 1:
                fl.plot_df(ax, phi_all, phi_sigma, hist_all, massBins=massBins, color=C, label='$\sigma = %d$'%sig, lw=2, alpha=0.95)

#     ## Ref
#     phi, phi_sigma, hist = fl.calc_df(mstar_ref[rtag] * 1e10, rtag, 100**3, massBinLimits)
#     fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C1', label='Ref (100 Mpc)', lw=2)
#     
#     ## AGN
#     phi, phi_sigma, hist = fl.calc_df(mstar_agn[rtag] * 1e10, rtag, 50**3, massBinLimits)
#     fl.plot_df(ax, phi, phi_sigma, hist, massBins, color='C2', label='AGNdT9 (50 Mpc)', lw=2)
 

    z = float(tag[5:].replace('p','.'))
    ax.text(0.1, 0.1, '$\mathrm{z = %.2f}$'%z, transform=ax.transAxes, size=12)
    # z = float(rtag[5:].replace('p','.'))
    # ax.text(0.6, 0.8, '$z (periodic) = %.2f$'%z, transform=ax.transAxes, size=12)
    
    ax.set_xlim(8, 11.8)
    ax.set_ylim(-7, -1.1) 
    ax.grid(alpha=0.3)
    

for ax in [ax2,ax3,ax5,ax6]:
    ax.set_yticklabels([])

for ax in [ax1,ax2,ax3]:
    ax.set_xticklabels([])
    
ax1.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax4.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax5.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)
ax2.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)

# ax1.legend(frameon=False, loc=1);
# ax6.legend()

sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []  # # fake up the array of the scalar mappable. Urgh...
cbar = plt.colorbar(sm, cax=cax, ticks=ticks, aspect=20)
cbar.ax.set_yticklabels(s_range)
cbar.ax.set_ylabel('$\sigma$', size=18, rotation=90)

# plt.show()
fig.savefig('images/gsmf_multi_split.png', dpi=200, bbox_inches='tight')

