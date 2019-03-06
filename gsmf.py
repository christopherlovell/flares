
import eagle as E
import pickle as pcl
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from misc import *
from scipy.spatial.distance import cdist

def plot_gsmf(mstar, tag, volume, ax, massBins, massBinLimits, 
              label, color, hist_lim=10, lw=3):
    
    hist, dummy = np.histogram(np.log10(mstar), bins = massBinLimits)
    hist = np.float64(hist)
    phi = (hist / volume) / (massBinLimits[1] - massBinLimits[0])
    phi_sigma = (np.sqrt(hist) / volume) / (massBinLimits[1] - massBinLimits[0]) # Poisson errors

    mask = (hist >= hist_lim)
    ax.plot(np.log10(massBins[mask][phi[mask] > 0.]), np.log10(phi[mask][phi[mask] > 0.]), 
            label=label, lw=lw, c=color, alpha=0.7)
    
    i = np.where(hist >= hist_lim)[0][-1]
    ax.plot(np.log10(massBins[i:][phi[i:] > 0.]), np.log10(phi[i:][phi[i:] > 0.]), 
            lw=lw, linestyle='dotted', c=color, alpha=0.7)
    
    return phi, phi_sigma, hist



massBinLimits = np.linspace(7.45, 13.25, 30)
massBins = np.logspace(7.55, 13.15, 29)
print(massBinLimits)
print(np.log10(massBins))

mstar = pcl.load(open('pcls/mstar.p','rb'))
sfr = pcl.load(open('pcls/sfr.p','rb'))
coods = pcl.load(open('pcls/coods.p','rb'))


gsmf_R = {halo: {} for halo in halos}
gsmf_mask = {halo: {} for halo in halos}

for halo in halos:
    for tag in tags:

        # median protocluster galaxy coordinate
        med_cood = np.median(coods[halo][tag], axis=0)

        # max distance of protocluster galaxy from median
        gsmf_R[halo][tag] = cdist([med_cood], coods[halo][tag]).max()

        print(halo, tag, gsmf_R[halo][tag])

        # mask all galaxies within this radius
        gsmf_mask[halo][tag] = cdist([med_cood], coods[halo][tag])[0] < gsmf_R[halo][tag]


# pcl.dump(gsmf_R, open('pcls/gsmf_R.p','wb'))
# pcl.dump(gsmf_mask, open('pcls/gsmf_mask.p','wb'))


#### Plot GSMF ####

fig = plt.figure(figsize=(9,10), dpi=150)

gs = gridspec.GridSpec(8, 3)
gs.update(wspace=0., hspace=0.)

ax1 = plt.subplot(gs[0:3, 0])
ax2 = plt.subplot(gs[0:3, 1])
ax3 = plt.subplot(gs[0:3, 2])
ax4 = plt.subplot(gs[4:7, 0])
ax5 = plt.subplot(gs[4:7, 1])
ax6 = plt.subplot(gs[4:7, 2])

axes = [ax1,ax2,ax3,ax4,ax5,ax6]

for ax, tag in zip(axes, tags): 

    z = float(tag[5:].replace('p','.'))
    scale_factor = 1. / (1+z)
    
    mstar_temp = np.hstack([mstar[halo][tag][gsmf_mask[halo][tag]] for halo in halos])
    mstar_temp = mstar_temp[mstar_temp > 0.] * 1e10

    V = np.sum([(4./3) * np.pi * (gsmf_R[halo][tag])**3 for halo in halos])
    
    phi_pc, dummy, dummy = plot_gsmf(mstar_temp, tag, V, ax, 
                                     massBins, massBinLimits, 'Protocluster', 'black')
    
    ax.text(0.1, 0.1, '$z = %.1f$'%z, transform=ax.transAxes, size=12)
    ax.set_xlim(8, 12.6)
    ax.set_ylim(-6, -.5) 
    ax.grid(alpha=0.5)
    

for ax in [ax2,ax3,ax5,ax6]:
    ax.set_yticklabels([])
    
# for ax in [ax1,ax2,ax3]:
#     ax.set_xticklabels([])
    
    
ax1.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax4.set_ylabel('$\mathrm{log_{10}}\,\phi$', size=13)
ax5.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)
ax2.set_xlabel('$\mathrm{log_{10}} \, (M_{*} \,/\, M_{\odot})$', size=13)

ax1.legend(frameon=False, loc=1);
ax6.legend()

plt.show()

# fig.savefig('output/gsmf.png', dpi=300, bbox_inches='tight')
