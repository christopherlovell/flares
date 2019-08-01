
import eagle as E
import numpy as np
# import scipy

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
mpl.rcParams['text.usetex'] = True

import fitDF.fitDF as fitDF
import fitDF.models as models
import fitDF.analyse as analyse

import flares

fl = flares.flares()


fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,12))

for i, (tag, rtag) in enumerate(zip(fl.tags[[5,6,7,8,9,10]], fl.ref_tags[[1,2,3,4,5,7]])):

    dz = .1

    if i==0: label='G-EAGLE'
    else: label=''
    a = analyse.analyse(ID='samples', sample_save_ID='geagle_gsmf_%s'%tag)
    c = 'C0'
    z = float(tag[5:].replace('p','.'))
    for ax, key in zip([ax1,ax2,ax3],a.parameters):
        ax.errorbar(z-dz, a.median_fit[key], c=c, fmt='o', label=label,
                 yerr=np.array([[np.percentile(a.samples[key], 16),
                       np.percentile(a.samples[key], 84)]])-a.median_fit[key])


    if i==0: label='Ref'
    else: label=''
    a = analyse.analyse(ID='samples', sample_save_ID='ref_gsmf_%s'%rtag)
    c = 'C1'
    z = float(tag[5:].replace('p','.'))
    for ax, key in zip([ax1,ax2,ax3],a.parameters):
        ax.errorbar(z, a.median_fit[key], c=c, fmt='o', label=label,
                 yerr=np.array([[np.percentile(a.samples[key], 16),
                       np.percentile(a.samples[key], 84)]])-a.median_fit[key])

    if i==0: label='AGNdT9'
    else: label=''
    a = analyse.analyse(ID='samples', sample_save_ID='agn_gsmf_%s'%rtag)
    c = 'C2'
    z = float(tag[5:].replace('p','.'))
    for ax, key in zip([ax1,ax2,ax3],a.parameters):
        ax.errorbar(z+dz, a.median_fit[key], c=c, fmt='o', label=label,
                 yerr=np.array([[np.percentile(a.samples[key], 16),
                       np.percentile(a.samples[key], 84)]])-a.median_fit[key])



ax3.set_xlabel('$z$',size=15)
ax1.set_ylabel('$\mathrm{log_{10}}(\\phi^{*})$',size=15)
ax2.set_ylabel('$\\alpha$',size=15)
ax3.set_ylabel('$M^{*}$',size=15)
ax3.legend()

for ax in [ax1,ax2,ax3]:
    ax.grid(alpha=0.3)

plt.savefig('images/fit_param_evolution.png',dpi=150)


